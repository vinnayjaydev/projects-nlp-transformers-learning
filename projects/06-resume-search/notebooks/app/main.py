# ==========================================================
# Resume Search API
# SBERT + FAISS + (Optional) Cross Encoder
# ==========================================================

from fastapi import FastAPI
from pydantic import BaseModel

import os
import pickle
import faiss
import numpy as np

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)

# ==========================================================
# FastAPI App
# ==========================================================

app = FastAPI(
    title="Resume Search API",
    description="AI ATS Resume Search using SBERT + FAISS",
    version="1.0.0",
)

# ==========================================================
# Paths
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIR = os.path.dirname(CURRENT_DIR)

INDEX_DIR = os.path.join(
    PROJECT_DIR,
    "indexes",
)

FAISS_INDEX_PATH = os.path.join(
    INDEX_DIR,
    "faiss_resume_index.bin",
)

METADATA_PATH = os.path.join(
    INDEX_DIR,
    "resume_metadata.pkl",
)

# ==========================================================
# Request Schema
# ==========================================================


class MatchRequest(BaseModel):

    job_description: str

    top_k: int = 5


# ==========================================================
# Load Models
# ==========================================================

print("=" * 60)
print("Loading Resume Search Resources...")
print("=" * 60)

# SBERT
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("✓ SBERT Model Loaded")

# Cross Encoder (optional)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("✓ Cross Encoder Loaded")

# FAISS
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

print("✓ FAISS Index Loaded")

# Metadata
with open(
    METADATA_PATH,
    "rb",
) as file:

    metadata = pickle.load(file)

print("✓ Metadata Loaded")
print("Metadata Keys:", metadata.keys())

print("✓ API Ready")
print("=" * 60)


# ==========================================================
# Retrieval Function
# ==========================================================


def retrieve_candidates(
    job_description,
    top_k=5,
):

    embedding = sbert_model.encode(job_description)

    embedding = np.array(
        [embedding],
        dtype="float32",
    )

    faiss.normalize_L2(embedding)

    scores, indices = faiss_index.search(
        embedding,
        top_k,
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):

        result = {
            "candidate": metadata["file_names"][idx],
            "faiss_score": round(
                float(score),
                4,
            ),
        }

        # Add resume text only if available
        if "resume_text" in metadata:

            result["resume_text"] = metadata["resume_text"][idx]

        results.append(result)

    return results


# ==========================================================
# Cross Encoder Re-ranking
# ==========================================================


def rerank_candidates(
    job_description,
    candidates,
):
    """
    If resume_text exists in metadata,
    use Cross Encoder reranking.

    Otherwise simply return FAISS results.
    """

    if len(candidates) == 0:
        return candidates

    if "resume_text" not in candidates[0]:
        return candidates

    pairs = []

    for candidate in candidates:

        pairs.append(
            (
                job_description,
                candidate["resume_text"],
            )
        )

    scores = cross_encoder.predict(pairs)

    for i in range(len(candidates)):

        candidates[i]["cross_score"] = float(scores[i])

    candidates = sorted(
        candidates,
        key=lambda x: x["cross_score"],
        reverse=True,
    )

    return candidates


# ==========================================================
# API Endpoints
# ==========================================================


@app.get("/")
def root():

    return {"message": "Resume Search API is running!"}


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "Resume Search API",
    }


@app.post("/match")
def match_candidates(
    request: MatchRequest,
):

    candidates = retrieve_candidates(
        request.job_description,
        request.top_k,
    )

    reranked = rerank_candidates(
        request.job_description,
        candidates,
    )

    return {
        "job_description": request.job_description,
        "num_results": len(reranked),
        "results": reranked,
    }
