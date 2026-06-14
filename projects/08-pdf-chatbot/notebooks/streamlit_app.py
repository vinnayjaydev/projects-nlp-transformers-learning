# ==========================================================
# Streamlit PDF Chatbot using RAG
# ==========================================================

import os
import tempfile

import faiss
import pdfplumber
import numpy as np
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

import torch

# ==========================================================
# Streamlit Configuration
# ==========================================================

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide",
)

st.title("📄 PDF Chatbot using RAG")

st.markdown("""
Upload a PDF and ask questions about its contents.

The chatbot retrieves relevant sections from the document
and generates answers using Retrieval-Augmented Generation (RAG).
""")

# ==========================================================
# Load Models
# ==========================================================


@st.cache_resource
def load_embedding_model():

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_llm():

    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=False,
        return_full_text=False,
    )

    return generator


embedding_model = load_embedding_model()
generator = load_llm()

# ==========================================================
# Session Memory
# ==========================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==========================================================
# PDF Upload
# ==========================================================

uploaded_pdf = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"],
)

# ==========================================================
# Save Uploaded PDF
# ==========================================================

pdf_path = None

if uploaded_pdf is not None:

    temp_dir = tempfile.mkdtemp()

    pdf_path = os.path.join(
        temp_dir,
        uploaded_pdf.name,
    )

    with open(
        pdf_path,
        "wb",
    ) as file:
        file.write(uploaded_pdf.read())

    st.sidebar.success("PDF Uploaded Successfully!")

# ==========================================================
# PDF Parsing
# ==========================================================


def parse_pdf(pdf_path):

    pages = []

    with pdfplumber.open(pdf_path) as pdf:

        for page_no, page in enumerate(
            pdf.pages,
            start=1,
        ):

            text = page.extract_text()

            if text:

                pages.append(
                    {
                        "page_number": page_no,
                        "text": text,
                    }
                )

    return pd.DataFrame(pages)


# ==========================================================
# Chunking
# ==========================================================


def chunk_document(
    pages_df,
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    records = []

    for _, row in pages_df.iterrows():

        text = str(row["text"]).strip()

        if text == "":
            continue

        chunks = splitter.split_text(text)

        for chunk in chunks:

            records.append(
                {
                    "page_number": row["page_number"],
                    "chunk_text": chunk,
                }
            )

    return pd.DataFrame(records)


# ==========================================================
# Embedding Generation
# ==========================================================


def create_embeddings(
    chunk_df,
):

    vectors = embedding_model.encode(
        chunk_df["chunk_text"].tolist(),
        convert_to_numpy=True,
    )

    vectors = np.array(
        vectors,
        dtype="float32",
    )

    faiss.normalize_L2(vectors)

    return vectors


# ==========================================================
# Build FAISS Index
# ==========================================================


def build_faiss_index(
    vectors,
):

    index = faiss.IndexFlatIP(vectors.shape[1])

    index.add(vectors)

    return index


# ==========================================================
# Retrieve Chunks
# ==========================================================


def retrieve_chunks(
    query,
    index,
    chunk_df,
    top_k=3,
):

    query_vector = embedding_model.encode(query)

    query_vector = np.array(
        [query_vector],
        dtype="float32",
    )

    faiss.normalize_L2(query_vector)

    scores, ids = index.search(
        query_vector,
        top_k,
    )

    retrieved = chunk_df.iloc[ids[0]].copy()

    retrieved["score"] = scores[0]

    return retrieved


# ==========================================================
# Prompt Builder
# ==========================================================


def build_prompt(
    query,
    retrieved_chunks,
    history,
):

    context = "\n\n".join(retrieved_chunks["chunk_text"].tolist())

    conversation = ""

    for message in history:

        conversation += f"{message['role']}: " f"{message['content']}\n"

    prompt = f"""
You are a helpful PDF assistant.

Answer ONLY using the supplied context.

If the answer cannot be found,
reply exactly:
"The information was not found in the document."

Conversation:
{conversation}

Context:
{context}

Question:
{query}

Answer:
"""

    return prompt


# ==========================================================
# RAG Pipeline
# ==========================================================


def ask_pdf(
    question,
    index,
    chunk_df,
):

    retrieved = retrieve_chunks(
        question,
        index,
        chunk_df,
    )

    prompt = build_prompt(
        question,
        retrieved,
        st.session_state.chat_history,
    )

    response = generator(prompt)

    answer = response[0]["generated_text"].strip()

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    return answer, retrieved


# ==========================================================
# Main Application
# ==========================================================

question = st.chat_input("Ask a question about the PDF...")

if uploaded_pdf is not None:

    pages_df = parse_pdf(pdf_path)

    chunk_df = chunk_document(pages_df)

    vectors = create_embeddings(chunk_df)

    index = build_faiss_index(vectors)

    st.sidebar.write(f"Pages: {len(pages_df)}")

    st.sidebar.write(f"Chunks: {len(chunk_df)}")

    if question:

        answer, retrieved = ask_pdf(
            question,
            index,
            chunk_df,
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            st.write(answer)

        st.subheader("🔍 Retrieved Context")

        for _, row in retrieved.iterrows():

            with st.expander(
                f"Page {row['page_number']} | Similarity: {row['score']:.3f}"
            ):

                st.write(row["chunk_text"])

# ==========================================================
# Conversation History
# ==========================================================

st.sidebar.subheader("🧠 Chat Memory")

for message in st.session_state.chat_history:

    st.sidebar.write(f"**{message['role']}**: " f"{message['content']}")
