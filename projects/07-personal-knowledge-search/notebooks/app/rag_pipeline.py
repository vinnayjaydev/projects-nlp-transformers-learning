"""
rag_pipeline.py

Pure PyTorch Local RAG Pipeline
(No TensorFlow / No Keras)
"""

import os

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from pathlib import Path
import pickle

import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

# ==========================================================
# Project Paths
# ==========================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

CHUNK_FILE = PROJECT_ROOT / "data" / "processed" / "chunked_corpus.csv"
FAISS_FILE = PROJECT_ROOT / "data" / "embeddings" / "faiss_index.bin"
EMBEDDING_FILE = PROJECT_ROOT / "data" / "embeddings" / "chunk_embeddings.pkl"

# ==========================================================
# Load Artifacts
# ==========================================================

print("=" * 60)
print("Loading Personal Knowledge Base...")
print("=" * 60)

chunks_df = pd.read_csv(CHUNK_FILE)

with open(EMBEDDING_FILE, "rb") as f:
    embeddings = pickle.load(f)

embeddings = np.array(embeddings, dtype="float32")

faiss_index = faiss.read_index(str(FAISS_FILE))

print("✓ Corpus Loaded")
print("✓ Embeddings Loaded")
print("✓ FAISS Index Loaded")

# ==========================================================
# Load SBERT
# ==========================================================

print("Loading SBERT Model...")

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("✓ SBERT Loaded")

# ==========================================================
# Retrieve Documents
# ==========================================================


def retrieve_documents(
    query: str,
    top_k: int = 3,
):

    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True,
    )

    query_embedding = np.array(
        [query_embedding],
        dtype="float32",
    )

    faiss.normalize_L2(query_embedding)

    scores, indices = faiss_index.search(
        query_embedding,
        top_k,
    )

    results = chunks_df.iloc[indices[0]].copy().reset_index(drop=True)

    results["score"] = scores[0]

    return results


# ==========================================================
# Build Prompt
# ==========================================================


def build_prompt(
    query,
    retrieved_docs,
):

    context = "\n\n".join(retrieved_docs["chunk_text"].tolist())

    prompt = f"""
Context:
{context}

Question:
{query}

Answer:
"""

    return prompt


# ==========================================================
# Lightweight Local "LLM"
# ==========================================================


def local_llm(
    prompt,
    retrieved_docs,
):
    """
    Simple placeholder.
    Replace later with Phi-3.
    """

    answer = "Based on the retrieved notes:\n\n" + "\n\n".join(
        retrieved_docs["chunk_text"].head(2).tolist()
    )

    return answer


# ==========================================================
# Main RAG Function
# ==========================================================


def personal_rag(
    query,
    top_k=3,
):

    retrieved_docs = retrieve_documents(
        query=query,
        top_k=top_k,
    )

    prompt = build_prompt(
        query=query,
        retrieved_docs=retrieved_docs,
    )

    answer = local_llm(
        prompt,
        retrieved_docs,
    )

    return {
        "query": query,
        "retrieved_docs": retrieved_docs,
        "prompt": prompt,
        "answer": answer,
    }


# ==========================================================
# Local Test
# ==========================================================

if __name__ == "__main__":

    result = personal_rag("How does semantic search work?")

    print(result["answer"])
