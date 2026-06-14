# ==========================================================
# Enterprise Search API
# Hybrid Search + ACL + FAISS + Cross Encoder
# ==========================================================

import ast
import re

import faiss
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rank_bm25 import BM25Okapi

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)

# ==========================================================
# Create FastAPI App
# ==========================================================

app = FastAPI(
    title="Enterprise Search API",
    description="Secure Hybrid Enterprise Search Engine",
    version="1.0.0",
)

# ==========================================================
# Load Enterprise Corpus
# ==========================================================

print("=" * 60)
print("Loading Enterprise Search Resources...")
print("=" * 60)

DATA_PATH = "data/processed/enterprise_corpus_acl.csv"

enterprise_df = pd.read_csv(
    DATA_PATH
)

# Convert ACL column from string to list
enterprise_df["acl_roles"] = (
    enterprise_df["acl_roles"].apply(
        lambda x:
        ast.literal_eval(x)
        if isinstance(x, str)
        else x
    )
)

# Create searchable text
enterprise_df["search_text"] = (
    enterprise_df["title"]
    + " "
    + enterprise_df["content"]
)

print("✓ Enterprise Corpus Loaded")

# ==========================================================
# User Profiles
# ==========================================================

USER_PROFILES = {
    "alice": {
        "roles": [
            "Engineering",
            "Public",
        ]
    },
    "bob": {
        "roles": [
            "HR",
            "Public",
        ]
    },
    "carol": {
        "roles": [
            "Finance",
            "Public",
        ]
    },
    "david": {
        "roles": [
            "Legal",
            "Public",
        ]
    },
}

# ==========================================================
# ACL Filter
# ==========================================================


def acl_filter(
    dataframe,
    user_roles,
):

    mask = dataframe[
        "acl_roles"
    ].apply(
        lambda acl:
        any(
            role in acl
            for role in user_roles
        )
    )

    return (
        dataframe[mask]
        .reset_index(drop=True)
    )


# ==========================================================
# Text Preprocessing
# ==========================================================


def preprocess_text(
    text,
):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s\-]",
        " ",
        text,
    )

    return text.split()


# ==========================================================
# Build BM25 Index
# ==========================================================

tokenized_corpus = [
    preprocess_text(doc)
    for doc in enterprise_df[
        "search_text"
    ]
]

bm25_model = BM25Okapi(
    tokenized_corpus
)

print("✓ BM25 Index Built")

# ==========================================================
# Load Embedding Models
# ==========================================================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("✓ Embedding Models Loaded")

# ==========================================================
# Build FAISS Index
# ==========================================================

document_embeddings = embedding_model.encode(
    enterprise_df[
        "search_text"
    ].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True,
)

document_embeddings = np.array(
    document_embeddings,
    dtype="float32",
)

faiss.normalize_L2(
    document_embeddings
)

faiss_index = faiss.IndexFlatIP(
    document_embeddings.shape[1]
)

faiss_index.add(
    document_embeddings
)

print("✓ FAISS Index Built")
print("✓ API Ready")
print("=" * 60)

# ==========================================================
# Candidate Retrieval
# ==========================================================


def retrieve_candidates(
    query,
    top_k=20,
):

    # BM25 Retrieval
    bm25_scores = bm25_model.get_scores(
        preprocess_text(
            query
        )
    )

    bm25_rank = np.argsort(
        bm25_scores
    )[::-1][:top_k]

    # Semantic Retrieval
    query_embedding = embedding_model.encode(
        query
    )

    query_embedding = np.array(
        [query_embedding],
        dtype="float32",
    )

    faiss.normalize_L2(
        query_embedding
    )

    _, semantic_rank = faiss_index.search(
        query_embedding,
        top_k,
    )

    # Hybrid Union
    candidates = list(
        set(
            list(
                bm25_rank
            )
            +
            list(
                semantic_rank[0]
            )
        )
    )

    return candidates


# ==========================================================
# Cross Encoder Re-ranking
# ==========================================================


def rerank_documents(
    query,
    candidate_indices,
    dataframe,
    top_k=5,
):

    if len(
        candidate_indices
    ) == 0:

        return []

    pairs = [

        (
            query,
            dataframe.iloc[idx][
                "search_text"
            ],
        )

        for idx in candidate_indices
    ]

    scores = cross_encoder.predict(
        pairs
    )

    results = []

    for idx, score in zip(
        candidate_indices,
        scores,
    ):

        row = dataframe.iloc[
            idx
        ].to_dict()

        row["score"] = float(
            score
        )

        results.append(
            row
        )

    results = sorted(
        results,
        key=lambda x:
        x["score"],
        reverse=True,
    )

    return results[
        :top_k
    ]


# ==========================================================
# API Schemas
# ==========================================================


class SearchRequest(
    BaseModel
):

    user_name: str

    query: str

    top_k: int = 5


class SearchResult(
    BaseModel
):

    doc_id: str

    title: str

    department: str

    score: float


# ==========================================================
# API Endpoints
# ==========================================================


@app.get("/")
def root():

    return {
        "message":
        "Enterprise Search API Running"
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service":
        "enterprise-search-api",
    }


@app.post("/search")
def search_documents(
    request: SearchRequest,
):

    # Validate User
    if (
        request.user_name
        not in USER_PROFILES
    ):

        raise HTTPException(
            status_code=404,
            detail="Unknown user.",
        )

    # ACL Filtering
    user_roles = USER_PROFILES[
        request.user_name
    ][
        "roles"
    ]

    authorized_df = acl_filter(
        enterprise_df,
        user_roles,
    )

    authorized_df = (
        authorized_df
        .reset_index(
            drop=True
        )
    )

    # Hybrid Retrieval
    candidate_docs = (
        retrieve_candidates(
            request.query,
            top_k=20,
        )
    )

    # Keep only valid indices
    candidate_docs = [

        idx

        for idx
        in candidate_docs

        if idx < len(
            authorized_df
        )

    ]

    # Re-ranking
    results = rerank_documents(
        request.query,
        candidate_docs,
        authorized_df,
        top_k=request.top_k,
    )

    return {
        "user":
        request.user_name,

        "query":
        request.query,

        "num_results":
        len(results),

        "results":
        results,
    }


# ==========================================================
# Run:
#
# uvicorn main:app --reload
#
# Swagger UI:
# http://127.0.0.1:8000/docs
#
# Health Check:
# http://127.0.0.1:8000/health
#
# Example POST:
#
# {
#   "user_name": "alice",
#   "query": "vector database migration",
#   "top_k": 3
# }
# ==========================================================