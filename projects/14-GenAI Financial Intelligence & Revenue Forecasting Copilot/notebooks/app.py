# ==========================================
# Financial Intelligence Copilot API
# ==========================================

from fastapi import FastAPI

from pydantic import BaseModel

import pandas as pd
import numpy as np

import faiss

from sentence_transformers import (
    SentenceTransformer
)

from xgboost import XGBRegressor


# ==========================================
# APP
# ==========================================

app = FastAPI(
    title="Financial Intelligence Copilot API"
)


# ==========================================
# LOAD DATA
# ==========================================

forecast_df = pd.read_csv(
    "../data/multimodal_forecasting_dataset.csv"
)

corpus_df = pd.read_csv(
    "../data/financial_rag_corpus.csv"
)


# ==========================================
# BUILD DOCUMENTS
# ==========================================

corpus_df["rag_document"] = (

    corpus_df["earnings_call"].fillna("")

    + " "

    + corpus_df["risk_factors"].fillna("")

    + " "

    + corpus_df["mda_section"].fillna("")
)


# ==========================================
# EMBEDDING MODEL
# ==========================================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)


# ==========================================
# DOCUMENT EMBEDDINGS
# ==========================================

document_embeddings = embedding_model.encode(

    corpus_df["rag_document"].tolist(),

    show_progress_bar=True
)


# ==========================================
# FAISS
# ==========================================

dimension = document_embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(

    np.array(
        document_embeddings
    ).astype(
        "float32"
    )
)


# ==========================================
# TRAIN MODEL
# ==========================================

feature_columns = [

    "gross_margin_pct",

    "operating_income_million",

    "net_income_million",

    "eps",

    "weighted_sentiment",

    "guidance_score",

    "positive_guidance_score",

    "total_risk_score",

    "normalized_risk_score",

    "macro_risk_score"
]

X = forecast_df[
    feature_columns
]

y = forecast_df[
    "revenue_million"
]

forecast_model = XGBRegressor(

    n_estimators=200,

    learning_rate=0.05,

    max_depth=4,

    random_state=42
)

forecast_model.fit(
    X,
    y
)


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance_df = pd.DataFrame({

    "feature":
    feature_columns,

    "importance":
    forecast_model.feature_importances_
})

importance_df = importance_df.sort_values(

    by="importance",

    ascending=False
)


# ==========================================
# REQUEST MODELS
# ==========================================

class ForecastRequest(
    BaseModel
):

    gross_margin_pct: float

    operating_income_million: float

    net_income_million: float

    eps: float

    weighted_sentiment: float

    guidance_score: float

    positive_guidance_score: float

    total_risk_score: float

    normalized_risk_score: float

    macro_risk_score: float


class SearchRequest(
    BaseModel
):

    query: str

    top_k: int = 3


# ==========================================
# RETRIEVAL FUNCTION
# ==========================================

def retrieve_documents(
    query,
    top_k=3
):

    query_embedding = embedding_model.encode(
        query
    )

    distances, indices = index.search(

        np.array(
            [query_embedding]
        ).astype(
            "float32"
        ),

        top_k
    )

    results = []

    for rank, idx in enumerate(
        indices[0]
    ):

        results.append({

            "rank":
            rank + 1,

            "quarter":
            corpus_df.iloc[idx][
                "quarter"
            ],

            "distance":
            float(
                distances[0][rank]
            ),

            "document":
            corpus_df.iloc[idx][
                "rag_document"
            ]
        })

    return results


# ==========================================
# HEALTH
# ==========================================

@app.get(
    "/health"
)
def health():

    return {
        "status": "healthy"
    }


# ==========================================
# FORECAST
# ==========================================

@app.post(
    "/forecast"
)
def forecast(
    request: ForecastRequest
):

    df = pd.DataFrame([{

        "gross_margin_pct":
        request.gross_margin_pct,

        "operating_income_million":
        request.operating_income_million,

        "net_income_million":
        request.net_income_million,

        "eps":
        request.eps,

        "weighted_sentiment":
        request.weighted_sentiment,

        "guidance_score":
        request.guidance_score,

        "positive_guidance_score":
        request.positive_guidance_score,

        "total_risk_score":
        request.total_risk_score,

        "normalized_risk_score":
        request.normalized_risk_score,

        "macro_risk_score":
        request.macro_risk_score
    }])

    prediction = forecast_model.predict(
        df
    )[0]

    return {

        "predicted_revenue_million":
        float(
            prediction
        )
    }


# ==========================================
# FORECAST EXPLANATION
# ==========================================

@app.post(
    "/forecast/explain"
)
def explain_forecast(
    request: ForecastRequest
):

    df = pd.DataFrame([{

        "gross_margin_pct":
        request.gross_margin_pct,

        "operating_income_million":
        request.operating_income_million,

        "net_income_million":
        request.net_income_million,

        "eps":
        request.eps,

        "weighted_sentiment":
        request.weighted_sentiment,

        "guidance_score":
        request.guidance_score,

        "positive_guidance_score":
        request.positive_guidance_score,

        "total_risk_score":
        request.total_risk_score,

        "normalized_risk_score":
        request.normalized_risk_score,

        "macro_risk_score":
        request.macro_risk_score
    }])

    prediction = forecast_model.predict(
        df
    )[0]

    top_features = importance_df.head(
        5
    )

    return {

        "predicted_revenue_million":
        float(
            prediction
        ),

        "top_drivers":
        top_features.to_dict(
            orient="records"
        )
    }


# ==========================================
# RAG SEARCH
# ==========================================

@app.post(
    "/rag/search"
)
def rag_search(
    request: SearchRequest
):

    results = retrieve_documents(

        request.query,

        request.top_k
    )

    return {

        "query":
        request.query,

        "results":
        results
    }


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

@app.get(
    "/feature-importance"
)
def feature_importance():

    return {

        "features":
        importance_df.to_dict(
            orient="records"
        )
    }