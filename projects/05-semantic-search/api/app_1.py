# ==========================================================
# api/app.py
# Semantic Search API
# ==========================================================

print("=" * 60)
print("LOADED APP FILE:")
print(__file__)
print("=" * 60)

from fastapi import FastAPI
from pydantic import BaseModel

import faiss
import pickle
import numpy as np

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)

# ==========================================================
# Create FastAPI App
# ==========================================================

app = FastAPI(
    title="Semantic Search API",
    description="Production Grade Semantic Search Engine",
    version="1.0.0",
)

# ==========================================================
# Load Models and Indexes
# ==========================================================

print("Loading models...")

# Bi-Encoder (Sentence Embeddings)
bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Cross Encoder (Re-ranking)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# FAISS Index
faiss_index = faiss.read_index("indexes/faiss_index.bin")

# Metadata
with open(
    "indexes/metadata.pkl",
    "rb",
) as file:
    metadata = pickle.load(file)

print("API Ready!")

# ==========================================================
# Request / Response Models
# ==========================================================


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    query: str
    results: list


# ==========================================================
# Retrieval Function
# ==========================================================


def retrieve_candidates(
    query: str,
    top_k: int = 5,
):
    # Convert query to embedding
    query_embedding = bi_encoder.encode(
        [query],
        convert_to_numpy=True,
    )

    # Search FAISS
    distances, indices = faiss_index.search(
        query_embedding.astype("float32"),
        top_k,
    )

    candidates = []

    for idx in indices[0]:
        candidates.append(metadata[idx])

    return candidates


# ==========================================================
# Re-ranking Function
# ==========================================================


def rerank_candidates(
    query: str,
    candidates: list,
):
    pairs = [(query, doc["text"]) for doc in candidates]

    scores = cross_encoder.predict(pairs)

    results = []

    for doc, score in zip(
        candidates,
        scores,
    ):
        results.append(
            {
                "title": doc["title"],
                "category": doc["category"],
                "score": float(score),
            }
        )

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True,
    )

    return results


# ==========================================================
# API Endpoints
# ==========================================================


@app.get("/")
def root():
    return {"message": "Semantic Search API is running!"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Semantic Search API",
    }


@app.post(
    "/search",
    response_model=SearchResponse,
)
def semantic_search(
    request: SearchRequest,
):
    candidates = retrieve_candidates(
        request.query,
        request.top_k,
    )

    ranked_results = rerank_candidates(
        request.query,
        candidates,
    )

    return {
        "query": request.query,
        "results": ranked_results,
    }
