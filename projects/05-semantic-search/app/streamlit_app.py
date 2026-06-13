# ==========================================================
# streamlit_app.py
# Production Grade Semantic Search Frontend
# Project: 05-semantic-search
# ==========================================================

import time
import requests
import pandas as pd
import streamlit as st

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(page_title="Semantic Search Engine", page_icon="🔍", layout="wide")


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("⚙️ Configuration")

API_URL = st.sidebar.text_input(
    "FastAPI Endpoint", value="http://127.0.0.1:8000/search"
)

TOP_K = st.sidebar.slider("Top K Results", min_value=1, max_value=10, value=5, step=1)

st.sidebar.markdown("---")

st.sidebar.subheader("💡 Example Queries")

example_queries = [
    "How do I learn Artificial Intelligence?",
    "Explain Machine Learning.",
    "Introduction to NLP.",
    "Deep Learning tutorials.",
    "What is a Transformer model?",
]

for example in example_queries:
    st.sidebar.write(f"• {example}")


# ==========================================================
# HEADER
# ==========================================================

st.title("🔍 Production Semantic Search Engine")

st.markdown("""
    This application demonstrates a **production-grade Semantic Search pipeline**
    using:

    - ✅ Sentence-BERT Embeddings
    - ✅ FAISS Vector Database
    - ✅ Hybrid Search (TF-IDF + SBERT)
    - ✅ Cross-Encoder Re-ranking
    - ✅ FastAPI Backend
    - ✅ Streamlit Frontend
    """)

st.divider()


# ==========================================================
# API CALL FUNCTION
# ==========================================================


@st.cache_data(show_spinner=False)
def call_search_api(query, top_k):

    payload = {"query": query, "top_k": top_k}

    response = requests.post(API_URL, json=payload, timeout=60)

    response.raise_for_status()

    return response.json()


# ==========================================================
# QUERY INPUT
# ==========================================================

query = st.text_area(
    label="Enter your search query",
    placeholder="Example: How can I learn Artificial Intelligence?",
    height=120,
)

col1, col2 = st.columns([1, 5])

with col1:
    search_button = st.button("🔎 Search", use_container_width=True)

with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.rerun()


# ==========================================================
# SEARCH EXECUTION
# ==========================================================

if search_button:

    if len(query.strip()) == 0:

        st.warning("⚠️ Please enter a search query.")

    else:

        try:

            with st.spinner("Searching documents..."):

                start_time = time.time()

                response = call_search_api(query=query, top_k=TOP_K)

                end_time = time.time()

                processing_time = round(end_time - start_time, 3)

            results = response.get("results", [])

            # ==================================================
            # METRICS
            # ==================================================

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(label="Documents Retrieved", value=len(results))

            metric2.metric(label="Processing Time", value=f"{processing_time} sec")

            metric3.metric(label="Search Type", value="Hybrid + CrossEncoder")

            st.divider()

            # ==================================================
            # RESULT TABLE
            # ==================================================

            if len(results) == 0:

                st.info("No matching documents found.")

            else:

                st.subheader("📋 Search Results")

                results_df = pd.DataFrame(results)

                results_df.index = results_df.index + 1

                st.dataframe(results_df, use_container_width=True)

                # ==============================================
                # SCORE CHART
                # ==============================================

                st.subheader("📊 Relevance Scores")

                chart_df = results_df.copy()

                if "title" in chart_df.columns:
                    chart_df = chart_df.set_index("title")

                    if "score" in chart_df.columns:

                        st.bar_chart(chart_df["score"])

                st.divider()

                # ==============================================
                # DETAILED RESULT CARDS
                # ==============================================

                st.subheader("📄 Detailed Results")

                for idx, row in results_df.iterrows():

                    title = row.get("title", "Unknown")

                    category = row.get("category", "N/A")

                    score = row.get("score", 0.0)

                    text = row.get("text", "Document preview unavailable.")

                    with st.expander(f"{idx}. {title}"):

                        st.markdown(f"**Category:** `{category}`")

                        st.markdown(f"**Relevance Score:** `{round(score, 3)}`")

                        st.markdown("**Document Preview:**")

                        st.write(text)

                # ==============================================
                # DOWNLOAD RESULTS
                # ==============================================

                csv_data = results_df.to_csv(index=False)

                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv_data,
                    file_name="semantic_search_results.csv",
                    mime="text/csv",
                )

        except requests.exceptions.ConnectionError:

            st.error("""
                ❌ Unable to connect to the FastAPI backend.

                Make sure your API server is running:

                uvicorn api.app:app --reload
                """)

        except requests.exceptions.Timeout:

            st.error("❌ API request timed out.")

        except Exception as e:

            st.error(f"❌ Unexpected Error: {str(e)}")


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown("""
    ### 🏗️ Architecture

    ```text
                      User
                        │
                        ▼
              Streamlit Frontend
                        │
                  HTTP POST /search
                        │
                        ▼
                  FastAPI Backend
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
      SBERT        FAISS Index   CrossEncoder
         │              │              │
         └──────────────┼──────────────┘
                        ▼
               Ranked Search Results
    ```

    **Pipeline:**  
    **Query → SBERT → FAISS Retrieval → Hybrid Search → Cross-Encoder Re-ranking → Final Results**
    """)
