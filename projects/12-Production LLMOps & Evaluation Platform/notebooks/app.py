import uuid
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# FastAPI Application
# =====================================================

app = FastAPI(
    title="LLMOps End-to-End Platform",
    version="1.0.0",
    description="Production LLMOps Monitoring & Evaluation Platform"
)

# =====================================================
# Create Required Directories
# =====================================================

Path("telemetry").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)
Path("evaluation_results").mkdir(exist_ok=True)

TRACE_FILE = "telemetry/final_platform_logs.csv"

# =====================================================
# Load Embedding Model
# =====================================================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# =====================================================
# Request Models
# =====================================================

class EvaluationRequest(BaseModel):

    question: str

    context: str

    generated_response: str

    ground_truth: str


class BatchEvaluationRequest(BaseModel):

    records: list


# =====================================================
# Response Model
# =====================================================

class EvaluationResponse(BaseModel):

    trace_id: str

    faithfulness: float

    answer_relevancy: float

    context_recall: float

    judge_score: float

    rag_health_score: float


# =====================================================
# Trace Generator
# =====================================================

def generate_trace_id():

    return str(uuid.uuid4())


# =====================================================
# Faithfulness Service
# Replace with Notebook 06
# =====================================================

def faithfulness_score():

    return 0.90


# =====================================================
# Relevancy Service
# Replace with Notebook 07
# =====================================================

def answer_relevancy():

    return 0.88


# =====================================================
# Recall Service
# Replace with Notebook 08
# =====================================================

def context_recall():

    return 0.87


# =====================================================
# LLM Judge Service
# Replace with Notebook 09
# =====================================================

def judge_score():

    return 0.91


# =====================================================
# Health Score Calculator
# =====================================================

def calculate_health_score(
    faithfulness,
    relevancy,
    recall,
    judge,
):

    score = (
        0.35 * faithfulness
        + 0.25 * relevancy
        + 0.20 * recall
        + 0.20 * judge
    )

    return round(score, 4)


# =====================================================
# Evaluation Pipeline
# =====================================================

def evaluate_response(
    question,
    context,
    answer,
    ground_truth,
):

    faithfulness = faithfulness_score()

    relevancy = answer_relevancy()

    recall = context_recall()

    judge = judge_score()

    rag_health = calculate_health_score(
        faithfulness,
        relevancy,
        recall,
        judge,
    )

    return {
        "faithfulness": faithfulness,
        "answer_relevancy": relevancy,
        "context_recall": recall,
        "judge_score": judge,
        "rag_health_score": rag_health,
    }


# =====================================================
# Telemetry Logger
# =====================================================

def log_trace(
    trace_id,
    question,
    metrics,
    latency_ms,
):

    trace = {
        "trace_id": trace_id,
        "timestamp": datetime.now(),
        "question": question,
        "latency_ms": latency_ms,
        **metrics,
    }

    df = pd.DataFrame([trace])

    file_exists = Path(TRACE_FILE).exists()

    df.to_csv(
        TRACE_FILE,
        mode="a",
        header=not file_exists,
        index=False,
    )

    return trace


# =====================================================
# Root Endpoint
# =====================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "service": "LLMOps Platform",
    }


# =====================================================
# Health Endpoint
# =====================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "timestamp": datetime.now(),
    }


# =====================================================
# Single Evaluation Endpoint
# =====================================================

@app.post(
    "/evaluate",
    response_model=EvaluationResponse,
)
def evaluate(
    request: EvaluationRequest,
):

    start_time = time.time()

    trace_id = generate_trace_id()

    metrics = evaluate_response(
        request.question,
        request.context,
        request.generated_response,
        request.ground_truth,
    )

    latency_ms = (
        time.time() - start_time
    ) * 1000

    log_trace(
        trace_id,
        request.question,
        metrics,
        latency_ms,
    )

    return {
        "trace_id": trace_id,
        **metrics,
    }


# =====================================================
# Batch Evaluation Endpoint
# =====================================================

@app.post("/evaluate/batch")
def batch_evaluate(
    request: BatchEvaluationRequest,
):

    results = []

    for record in request.records:

        metrics = evaluate_response(
            record["question"],
            record["context"],
            record["generated_response"],
            record["ground_truth"],
        )

        results.append(metrics)

    return {
        "records": len(results),
        "results": results,
    }


# =====================================================
# Telemetry Endpoint
# =====================================================

@app.get("/telemetry")
def telemetry():

    if not Path(TRACE_FILE).exists():

        return []

    df = pd.read_csv(TRACE_FILE)

    return df.tail(20).to_dict(
        orient="records"
    )


# =====================================================
# Dashboard Endpoint
# =====================================================

@app.get("/dashboard")
def dashboard():

    if not Path(TRACE_FILE).exists():

        return {
            "total_requests": 0,
            "avg_faithfulness": 0,
            "avg_relevancy": 0,
            "avg_recall": 0,
            "avg_judge": 0,
            "avg_health": 0,
        }

    df = pd.read_csv(TRACE_FILE)

    return {
        "total_requests": len(df),

        "avg_faithfulness": round(
            df["faithfulness"].mean(),
            3,
        ),

        "avg_relevancy": round(
            df["answer_relevancy"].mean(),
            3,
        ),

        "avg_recall": round(
            df["context_recall"].mean(),
            3,
        ),

        "avg_judge": round(
            df["judge_score"].mean(),
            3,
        ),

        "avg_health": round(
            df["rag_health_score"].mean(),
            3,
        ),
    }


# =====================================================
# Drift Monitoring Endpoint
# =====================================================

@app.get("/drift")
def drift_monitor():

    drift_file = (
        "evaluation_results/drift_monitoring.csv"
    )

    if not Path(drift_file).exists():

        return {
            "status": "No Drift Data Available"
        }

    drift_df = pd.read_csv(
        drift_file
    )

    latest_drift = drift_df[
        "distance"
    ].iloc[-1]

    return {
        "latest_drift": float(
            latest_drift
        ),
        "alert": latest_drift > 0.25,
    }


# =====================================================
# Platform Metrics Endpoint
# =====================================================

@app.get("/platform-metrics")
def platform_metrics():

    telemetry_records = []

    for _ in range(100):

        metrics = {
            "faithfulness":
                np.random.uniform(
                    0.7,
                    1.0
                ),

            "answer_relevancy":
                np.random.uniform(
                    0.7,
                    1.0
                ),

            "context_recall":
                np.random.uniform(
                    0.7,
                    1.0
                ),

            "judge_score":
                np.random.uniform(
                    0.7,
                    1.0
                ),
        }

        metrics[
            "rag_health_score"
        ] = calculate_health_score(
            metrics["faithfulness"],
            metrics["answer_relevancy"],
            metrics["context_recall"],
            metrics["judge_score"],
        )

        telemetry_records.append(
            metrics
        )

    telemetry_df = pd.DataFrame(
        telemetry_records
    )

    dashboard_metrics = {
        "total_requests":
            len(telemetry_df),

        "avg_faithfulness":
            round(
                telemetry_df[
                    "faithfulness"
                ].mean(),
                3,
            ),

        "avg_relevancy":
            round(
                telemetry_df[
                    "answer_relevancy"
                ].mean(),
                3,
            ),

        "avg_recall":
            round(
                telemetry_df[
                    "context_recall"
                ].mean(),
                3,
            ),

        "avg_judge":
            round(
                telemetry_df[
                    "judge_score"
                ].mean(),
                3,
            ),

        "avg_health":
            round(
                telemetry_df[
                    "rag_health_score"
                ].mean(),
                3,
            ),
    }

    return dashboard_metrics


# =====================================================
# Run Application
# =====================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )