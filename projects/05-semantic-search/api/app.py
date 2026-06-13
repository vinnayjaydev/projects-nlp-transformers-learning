# ==========================================================
# Semantic Search API
# File: api/app.py
# ==========================================================

from pathlib import Path
import pickle

import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)

# ==========================================================
# Create FastAPI Application
# ==========================================================

app = FastAPI(
    title="Semantic Search API",
    description="Production Grade Semantic Search Engine using SBERT + FAISS + Cross Encoder",
    version="1.0.0",
)

# ==========================================================
# Global Variables
# ==========================================================

bi_encoder = None
cross_encoder = None
faiss_index = None
metadata = None

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================================================
# Load Models and Data on Startup
# ==========================================================


@app.on_event("startup")
def load_resources():
    global bi_encoder
    global cross_encoder
    global faiss_index
    global metadata

    print("=" * 60)
    print("Loading Semantic Search Resources...")
    print("=" * 60)

    # ------------------------------------------------------
    # Load Sentence Transformer (Bi-Encoder)
    # ------------------------------------------------------
    bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # ------------------------------------------------------
    # Load Cross Encoder
    # ------------------------------------------------------
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ------------------------------------------------------
    # Load FAISS Index
    # ------------------------------------------------------
    index_path = BASE_DIR / "indexes" / "faiss_index.bin"

    faiss_index = faiss.read_index(str(index_path))

    # ------------------------------------------------------
    # Load Metadata
    # ------------------------------------------------------
    metadata_path = BASE_DIR / "indexes" / "metadata.pkl"

    with open(metadata_path, "rb") as file:
        metadata = pickle.load(file)

    print("✓ Models Loaded")
    print("✓ FAISS Index Loaded")
    print("✓ Metadata Loaded")
    print("✓ API Ready")
    print("=" * 60)


# ==========================================================
# Request / Response Models
# ==========================================================


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    title: str
    category: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


# ==========================================================
# Candidate Retrieval using FAISS
# ==========================================================


def retrieve_candidates(
    query: str,
    top_k: int = 5,
):
    """
    Retrieve top-k candidate documents
    using SBERT embeddings + FAISS.
    """

    # Generate embedding for query
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
        if idx != -1:
            candidates.append(metadata[idx])

    return candidates


# ==========================================================
# Re-rank using Cross Encoder
# ==========================================================


def rerank_candidates(
    query: str,
    candidates: list,
):
    """
    Re-rank FAISS retrieved documents
    using a Cross Encoder.
    """

    if len(candidates) == 0:
        return []

    # Create (query, document) pairs
    pairs = [(query, doc["text"]) for doc in candidates]

    # Predict relevance scores
    scores = cross_encoder.predict(pairs)

    results = []

    for doc, score in zip(
        candidates,
        scores,
    ):
        results.append(
            SearchResult(
                title=doc["title"],
                category=doc["category"],
                score=float(score),
            )
        )

    # Sort descending by score
    results = sorted(
        results,
        key=lambda x: x.score,
        reverse=True,
    )

    return results


# ==========================================================
# API Endpoints
# ==========================================================


@app.get("/")
def root():
    """
    Root endpoint.
    """

    return {
        "message": "Semantic Search API is running!",
        "version": "1.0.0",
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """

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
    """
    Semantic Search Endpoint
    """

    # Retrieve candidate documents
    candidates = retrieve_candidates(
        request.query,
        request.top_k,
    )

    # Re-rank retrieved documents
    ranked_results = rerank_candidates(
        request.query,
        candidates,
    )

    # Return response
    return SearchResponse(
        query=request.query,
        results=ranked_results,
    )
