# ==========================================
# AI Powered Insurance Fraud Investigation
# FastAPI Service
# ==========================================

from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import os

# ==========================================
# Data Location
# ==========================================

DATA_DIR = r"D:\Books\projects-nlp-transformers-learning\projects\13-AI-Powered Insurance Fraud Investigation Assistant\data"

# ==========================================
# Load Data
# ==========================================

claims_df = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "insurance_claims.csv"
    )
)

risk_df = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "fraud_risk_scores.csv"
    )
)

brief_df = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "llm_investigation_briefs.csv"
    )
)

# ==========================================
# FastAPI App
# ==========================================

app = FastAPI(

    title="Insurance Fraud Investigation API",

    version="1.0.0",

    description="""
    Insurance Fraud Detection
    Investigation Platform
    """
)

# ==========================================
# Request Models
# ==========================================

class ClaimRequest(BaseModel):

    claim_id: str


# ==========================================
# Root Endpoint
# ==========================================

@app.get("/")
def home():

    return {

        "message":
        "Insurance Fraud Investigation API",

        "version":
        "1.0.0"
    }


# ==========================================
# Get All Claims
# ==========================================

@app.get("/claims")
def get_claims():

    return claims_df.to_dict(
        orient="records"
    )


# ==========================================
# High Risk Claims
# ==========================================

@app.get("/high-risk-claims")
def high_risk_claims():

    results = risk_df[

        risk_df[
            "fraud_risk_index"
        ] >= 60

    ]

    return results.to_dict(
        orient="records"
    )


# ==========================================
# Claim Details
# ==========================================

@app.get("/claim/{claim_id}")
def get_claim(
    claim_id: str
):

    result = claims_df[

        claims_df[
            "claim_id"
        ] == claim_id

    ]

    if len(result) == 0:

        return {

            "error":
            "Claim not found"
        }

    return result.iloc[0].to_dict()


# ==========================================
# Risk Score
# ==========================================

@app.post("/risk-score")
def get_risk_score(
    request: ClaimRequest
):

    result = risk_df[

        risk_df[
            "claim_id"
        ] == request.claim_id

    ]

    if len(result) == 0:

        return {

            "error":
            "Claim not found"
        }

    return result.iloc[0].to_dict()


# ==========================================
# Investigation Brief
# ==========================================

@app.get("/investigation-brief/{claim_id}")
def get_investigation_brief(
    claim_id: str
):

    result = brief_df[

        brief_df[
            "claim_id"
        ] == claim_id

    ]

    if len(result) == 0:

        return {

            "error":
            "Brief not found"
        }

    return result.iloc[0].to_dict()


# ==========================================
# Generate Brief
# ==========================================

@app.post("/generate-brief")
def generate_brief(
    request: ClaimRequest
):

    result = brief_df[

        brief_df[
            "claim_id"
        ] == request.claim_id

    ]

    if len(result) == 0:

        return {

            "error":
            "Claim not found"
        }

    return {

        "claim_id":
        request.claim_id,

        "brief":
        result.iloc[0].to_dict()
    }


# ==========================================
# Full Investigation
# ==========================================

@app.post("/investigate")
def investigate(
    request: ClaimRequest
):

    claim = claims_df[

        claims_df[
            "claim_id"
        ] == request.claim_id

    ]

    risk = risk_df[

        risk_df[
            "claim_id"
        ] == request.claim_id

    ]

    brief = brief_df[

        brief_df[
            "claim_id"
        ] == request.claim_id

    ]

    if len(claim) == 0:

        return {

            "error":
            "Claim not found"
        }

    response = {

        "claim":
        claim.iloc[0].to_dict()
    }

    if len(risk) > 0:

        response["risk"] = (

            risk.iloc[0].to_dict()
        )

    if len(brief) > 0:

        response["investigation_brief"] = (

            brief.iloc[0].to_dict()
        )

    return response


# ==========================================
# Health Check
# ==========================================

@app.get("/health")
def health():

    return {

        "status":
        "healthy"
    }