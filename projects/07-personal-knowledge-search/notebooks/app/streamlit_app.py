"""
streamlit_app.py

Personal AI Second Brain
------------------------
Frontend UI for Local RAG over your personal notes,
books, journals, and project documents.
"""

import streamlit as st
import pandas as pd

from rag_pipeline import personal_rag

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Personal AI Second Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# Header
# ==========================================================

st.title("🧠 Personal AI Second Brain")

st.markdown("""
Search your **journals, books, projects, notes, and research papers**
using **Semantic Search + FAISS + Local RAG + Phi-3 Mini**.

Ask questions naturally and let your personal knowledge base answer them.
""")

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("⚙️ Search Settings")

search_mode = st.sidebar.selectbox(
    "Search Mode",
    [
        "Hybrid Search",
        "Semantic Only",
    ],
)

category = st.sidebar.selectbox(
    "Category",
    [
        "All",
        "project",
        "journal",
        "book",
        "notes",
    ],
)

top_k = st.sidebar.slider(
    "Top K Documents",
    min_value=1,
    max_value=10,
    value=3,
)

show_scores = st.sidebar.checkbox(
    "Show Retrieval Scores",
    value=True,
)

show_prompt = st.sidebar.checkbox(
    "Show Generated Prompt",
    value=False,
)

# ==========================================================
# User Query
# ==========================================================

query = st.text_input(
    "🔎 Ask your Second Brain",
    placeholder="Example: How does semantic search work?",
)

# ==========================================================
# Execute RAG
# ==========================================================

if query:

    with st.spinner("🔍 Searching your knowledge base..."):

        try:
            output = personal_rag(
                query=query,
                top_k=top_k,
            )

        except Exception as e:
            st.error(f"An error occurred:\n\n{e}")
            st.stop()

    # ======================================================
    # AI Answer
    # ======================================================

    st.subheader("🤖 AI Answer")

    st.success(output.get("answer", "No answer generated."))

    # ======================================================
    # Retrieved Documents
    # ======================================================

    st.subheader("📄 Retrieved Documents")

    retrieved_docs = output.get(
        "retrieved_docs",
        pd.DataFrame(),
    )

    if len(retrieved_docs) == 0:

        st.warning("No matching documents were found.")

    else:

        for _, row in retrieved_docs.iterrows():

            source = row.get(
                "source",
                "Unknown",
            )

            score = row.get(
                "score",
                0.0,
            )

            chunk_text = row.get(
                "chunk_text",
                "",
            )

            with st.expander(f"📌 {source}  |  Score: {score:.3f}"):
                st.write(chunk_text)

    # ======================================================
    # Related Thoughts
    # ======================================================

    if len(retrieved_docs) > 0:

        st.subheader("🕸️ Related Thoughts")

        related_sources = retrieved_docs["source"].drop_duplicates().tolist()

        for item in related_sources:
            st.markdown(f"- {item}")

    # ======================================================
    # Similarity Score Chart
    # ======================================================

    if show_scores and len(retrieved_docs) > 0 and "score" in retrieved_docs.columns:

        st.subheader("📊 Retrieval Scores")

        chart_df = (
            retrieved_docs[["source", "score"]].drop_duplicates().set_index("source")
        )

        st.bar_chart(chart_df)

    # ======================================================
    # Optional: Show Generated Prompt
    # ======================================================

    if show_prompt:

        st.subheader("📝 Generated RAG Prompt")

        st.code(
            output.get("prompt", ""),
            language="text",
        )

# ==========================================================
# Footer
# ==========================================================

st.markdown("---")

st.caption(
    "🚀 Built with Streamlit • Sentence Transformers • FAISS • Local RAG • Phi-3 Mini"
)
