# =====================================================
# Streamlit Enterprise RAG Assistant
# =====================================================

import streamlit as st
import requests

# -----------------------------------------------------
# Page Config
# -----------------------------------------------------

st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------------------------------
# Header
# -----------------------------------------------------

st.title("🤖 Enterprise Multi-Document RAG Assistant")

st.markdown(
    """
Ask questions across:

- HR Documents
- Finance Reports
- Engineering Roadmaps

Powered by:

- Semantic Router
- BM25 Retrieval
- Vector Search
- Cross Encoder Reranking
- FastAPI Backend
"""
)

# -----------------------------------------------------
# Session State
# -----------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------------------------------
# User Input
# -----------------------------------------------------

question = st.text_input(
    "Ask a question",
    placeholder="What AI infrastructure projects are planned?"
)

# -----------------------------------------------------
# Ask Button
# -----------------------------------------------------

if st.button("Submit Question"):

    if question.strip():

        try:

            response = requests.post(
                "http://127.0.0.1:8000/ask",
                json={
                    "question": question
                },
                timeout=60
            )

            result = response.json()

            answer = result.get(
                "answer",
                "No answer returned."
            )

            sources = result.get(
                "sources",
                0
            )

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": answer,
                    "sources": sources
                }
            )

        except Exception as e:

            st.error(
                f"Connection Error: {e}"
            )

# -----------------------------------------------------
# Display Chat
# -----------------------------------------------------

for item in reversed(
    st.session_state.chat_history
):

    st.markdown("---")

    st.markdown(
        f"### 🙋 Question\n{item['question']}"
    )

    st.markdown(
        f"### 🤖 Answer\n{item['answer']}"
    )

    st.info(
        f"Retrieved Sources: {item['sources']}"
    )