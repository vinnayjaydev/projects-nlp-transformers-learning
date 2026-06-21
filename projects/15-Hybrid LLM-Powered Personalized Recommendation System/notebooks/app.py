# ==========================================
# Recommendation API
# app.py
# ==========================================

from fastapi import FastAPI

import pandas as pd

# ==========================================
# Load Data
# ==========================================

profiles_df = pd.read_csv(
    "../data/item_profiles.csv"
)

recommendations_df = pd.read_csv(
    "../data/hybrid_recommendations.csv"
)

evaluation_df = pd.read_csv(
    "../data/recommendation_evaluation.csv"
)

summary_df = pd.read_csv(
    "../data/recommendation_summary_metrics.csv"
)

# ==========================================
# FastAPI App
# ==========================================

app = FastAPI(
    title="Hybrid Recommendation API",
    version="1.0"
)

# ==========================================
# Home
# ==========================================

@app.get("/")
def home():

    return {
        "project":
        "Hybrid Recommendation System",

        "status":
        "running"
    }

# ==========================================
# Recommend User
# ==========================================

@app.get("/recommend/{user_id}")
def recommend_user(
    user_id: int
):

    user_recommendations = recommendations_df[

        recommendations_df[
            "user_id"
        ] == user_id

    ]

    return {

        "user_id":
        user_id,

        "recommendations":
        user_recommendations.to_dict(
            orient="records"
        )
    }

# ==========================================
# Similar Item
# ==========================================

@app.get("/similar-items/{item_id}")
def similar_item(
    item_id: int
):

    item = profiles_df[

        profiles_df[
            "item_id"
        ] == item_id

    ]

    if len(item) == 0:

        return {
            "message":
            "Item not found"
        }

    category = item.iloc[0][
        "category"
    ]

    similar_items = profiles_df[

        profiles_df[
            "category"
        ] == category

    ]

    return {

        "item_id":
        item_id,

        "category":
        category,

        "similar_items":
        similar_items.to_dict(
            orient="records"
        )
    }

# ==========================================
# Cold Start
# ==========================================

@app.get("/cold-start")
def cold_start():

    popular_items = profiles_df.head(
        10
    )

    return {

        "strategy":
        "popular_items",

        "items":
        popular_items.to_dict(
            orient="records"
        )
    }

# ==========================================
# Evaluation Metrics
# ==========================================

@app.get("/evaluation")
def evaluation():

    return {

        "metrics":

        summary_df.to_dict(
            orient="records"
        )
    }

# ==========================================
# User Evaluation
# ==========================================

@app.get("/evaluation/{user_id}")
def user_evaluation(
    user_id: int
):

    user_metrics = evaluation_df[

        evaluation_df[
            "user_id"
        ] == user_id

    ]

    return {

        "user_id":
        user_id,

        "metrics":

        user_metrics.to_dict(
            orient="records"
        )
    }

# ==========================================
# Catalog
# ==========================================

@app.get("/catalog")
def catalog():

    return profiles_df.to_dict(
        orient="records"
    )

# ==========================================
# Health Check
# ==========================================

@app.get("/health")
def health():

    return {
        "status":
        "healthy"
    }