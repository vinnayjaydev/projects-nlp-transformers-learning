# app.py


from fastapi import FastAPI
from pydantic import BaseModel

import pickle
import pandas as pd
import numpy as np
import faiss

from rank_bm25 import BM25Okapi
from transformers import pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# FastAPI App
# =====================================================

app = FastAPI(
    title="Enterprise RAG API",
    version="1.0.0",
    description="Enterprise Multi-Document RAG Assistant",
)

# =====================================================
# Request Models
# =====================================================

class QuestionRequest(BaseModel):
    question: str


# =====================================================
# Load Models
# =====================================================

print("Loading models...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

print("Models loaded.")

# =====================================================
# Load Data
# =====================================================

children_df = pd.read_csv(
    "artifacts/child_chunks.csv"
)

parents_df = pd.read_csv(
    "artifacts/parent_chunks.csv"
)

metadata_df = pd.read_csv(
    "artifacts/document_metadata.csv"
)

children_df = children_df.merge(
    metadata_df[
        ["document_id", "department"]
    ],
    on="document_id",
    how="left",
)

# =====================================================
# Router Embeddings
# =====================================================

with open(
    "artifacts/router_embeddings.pkl",
    "rb",
) as file:

    department_embeddings = pickle.load(
        file
    )

# =====================================================
# FAISS Indexes
# =====================================================

vector_indexes = {}

for dept in [
    "hr",
    "finance",
    "engineering",
]:

    vector_indexes[dept] = faiss.read_index(
        f"artifacts/faiss/{dept}.index"
    )

# =====================================================
# Chunk Mappings
# =====================================================

with open(
    "artifacts/faiss/chunk_mapping.pkl",
    "rb",
) as file:

    chunk_mappings = pickle.load(
        file
    )

# =====================================================
# BM25
# =====================================================

department_bm25 = {}
department_docs = {}

for dept in children_df["department"].unique():

    subset = children_df[
        children_df["department"] == dept
    ].reset_index(drop=True)

    corpus = [
        text.lower().split()
        for text in subset["content"]
    ]

    department_bm25[dept] = BM25Okapi(
        corpus
    )

    department_docs[dept] = subset

# =====================================================
# Conversation Memory
# =====================================================

conversation_memory = []

# =====================================================
# Query Rewriter
# =====================================================

def rewrite_query(question):

    history = "\n".join(
        [
            x["content"]
            for x in conversation_memory[-6:]
        ]
    )

    prompt = f"""
Rewrite into standalone question.

History:
{history}

Question:
{question}
"""

    response = llm(
        prompt,
        max_new_tokens=64,
    )

    return response[0]["generated_text"]

# =====================================================
# Router
# =====================================================

def route_query(query):

    query_embedding = embedding_model.encode(
        query
    )

    scores = {}

    for dept, emb in department_embeddings.items():

        score = cosine_similarity(
            query_embedding.reshape(1, -1),
            emb.reshape(1, -1),
        )[0][0]

        scores[dept] = score

    return max(
        scores,
        key=scores.get,
    )

# =====================================================
# BM25 Search
# =====================================================

def bm25_search(
    query,
    department,
    top_k=20,
):

    docs = department_docs[
        department
    ].copy()

    scores = department_bm25[
        department
    ].get_scores(
        query.lower().split()
    )

    docs["score"] = scores

    docs = docs.sort_values(
        "score",
        ascending=False,
    )

    return docs.head(top_k)

# =====================================================
# Semantic Search
# =====================================================

def semantic_search(
    query,
    department,
    top_k=20,
):

    query_embedding = embedding_model.encode(
        [query]
    ).astype("float32")

    faiss.normalize_L2(
        query_embedding
    )

    scores, ids = vector_indexes[
        department
    ].search(
        query_embedding,
        top_k,
    )

    mapping = chunk_mappings[
        department
    ]

    results = []

    for score, idx in zip(
        scores[0],
        ids[0],
    ):

        row = mapping.iloc[idx]

        results.append(
            {
                "parent_id": row["parent_id"],
                "content": row["content"],
                "score": float(score),
            }
        )

    return results

# =====================================================
# Hybrid Candidates
# =====================================================

def hybrid_candidates(query):

    department = route_query(query)

    bm25_docs = bm25_search(
        query,
        department,
    )

    semantic_docs = semantic_search(
        query,
        department,
    )

    candidates = {}

    for _, row in bm25_docs.iterrows():

        candidates[
            row["parent_id"]
        ] = row["content"]

    for row in semantic_docs:

        candidates[
            row["parent_id"]
        ] = row["content"]

    return list(
        candidates.items()
    )

# =====================================================
# Reranking
# =====================================================

def rerank_documents(
    query,
    candidates,
    top_k=5,
):

    pairs = [
        [query, text]
        for _, text in candidates
    ]

    scores = cross_encoder.predict(
        pairs
    )

    ranked = []

    for (
        doc_id,
        text,
    ), score in zip(
        candidates,
        scores,
    ):

        ranked.append(
            {
                "parent_id": doc_id,
                "content": text,
                "score": float(score),
            }
        )

    ranked = sorted(
        ranked,
        key=lambda x: x["score"],
        reverse=True,
    )

    return ranked[:top_k]

# =====================================================
# Context Builder
# =====================================================

def build_context(ranked_docs):

    return "\n\n".join(
        [
            doc["content"]
            for doc in ranked_docs
        ]
    )

# =====================================================
# Answer Generation
# =====================================================

def generate_answer(
    question,
    context,
):

    prompt = f"""
Answer only from context.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm(
        prompt,
        max_new_tokens=256,
    )

    return response[0]["generated_text"]

# =====================================================
# Main RAG Pipeline
# =====================================================

def rag_pipeline(question):

    standalone_question = rewrite_query(
        question
    )

    candidates = hybrid_candidates(
        standalone_question
    )

    ranked_docs = rerank_documents(
        standalone_question,
        candidates,
    )

    context = build_context(
        ranked_docs
    )

    answer = generate_answer(
        standalone_question,
        context,
    )

    conversation_memory.append(
        {
            "role": "user",
            "content": question,
        }
    )

    conversation_memory.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    return {
        "answer": answer,
        "sources": len(ranked_docs),
    }

# =====================================================
# API Endpoints
# =====================================================

@app.post("/ask")
def ask_question(
    request: QuestionRequest,
):

    return rag_pipeline(
        request.question
    )

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }

