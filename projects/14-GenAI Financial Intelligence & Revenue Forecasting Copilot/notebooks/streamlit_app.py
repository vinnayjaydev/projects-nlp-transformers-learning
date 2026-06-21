# ==========================================
# Financial Intelligence Copilot Dashboard
# ==========================================

import streamlit as st

import pandas as pd
import numpy as np

import requests

import plotly.express as px
import plotly.graph_objects as go


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(

    page_title="Financial Intelligence Copilot",

    page_icon="📈",

    layout="wide"
)


# ==========================================
# API URL
# ==========================================

API_URL = "http://127.0.0.1:8000"


# ==========================================
# TITLE
# ==========================================

st.title(
    "📈 Financial Intelligence & Revenue Forecasting Copilot"
)

st.markdown("---")


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title(
    "Financial Controls"
)

gross_margin = st.sidebar.slider(

    "Gross Margin %",

    0,

    100,

    65
)

operating_income = st.sidebar.number_input(

    "Operating Income (Million)",

    value=35.0
)

net_income = st.sidebar.number_input(

    "Net Income (Million)",

    value=28.0
)

eps = st.sidebar.number_input(

    "EPS",

    value=2.0
)

weighted_sentiment = st.sidebar.slider(

    "Sentiment",

    -1.0,

    1.0,

    0.8
)

guidance_score = st.sidebar.slider(

    "Guidance Score",

    0,

    10,

    4
)

positive_guidance_score = st.sidebar.slider(

    "Positive Guidance Score",

    0,

    10,

    3
)

total_risk_score = st.sidebar.slider(

    "Total Risk Score",

    0,

    20,

    2
)

normalized_risk_score = st.sidebar.slider(

    "Normalized Risk Score",

    0.0,

    1.0,

    0.25
)

macro_risk_score = st.sidebar.slider(

    "Macro Risk Score",

    0.0,

    1.0,

    0.30
)


# ==========================================
# REQUEST PAYLOAD
# ==========================================

payload = {

    "gross_margin_pct":
    gross_margin,

    "operating_income_million":
    operating_income,

    "net_income_million":
    net_income,

    "eps":
    eps,

    "weighted_sentiment":
    weighted_sentiment,

    "guidance_score":
    guidance_score,

    "positive_guidance_score":
    positive_guidance_score,

    "total_risk_score":
    total_risk_score,

    "normalized_risk_score":
    normalized_risk_score,

    "macro_risk_score":
    macro_risk_score
}


# ==========================================
# TABS
# ==========================================

tab1, tab2, tab3, tab4 = st.tabs([

    "Revenue Forecast",

    "Forecast Explanation",

    "Financial RAG",

    "Feature Importance"
])


# ==========================================
# TAB 1
# ==========================================

with tab1:

    st.header(
        "Revenue Forecast"
    )

    if st.button(
        "Generate Forecast"
    ):

        response = requests.post(

            f"{API_URL}/forecast",

            json=payload
        )

        result = response.json()

        prediction = result[
            "predicted_revenue_million"
        ]

        st.metric(

            "Predicted Revenue",

            f"${prediction:,.2f}M"
        )


# ==========================================
# TAB 2
# ==========================================

with tab2:

    st.header(
        "Forecast Explanation"
    )

    if st.button(
        "Explain Forecast"
    ):

        response = requests.post(

            f"{API_URL}/forecast/explain",

            json=payload
        )

        result = response.json()

        st.metric(

            "Predicted Revenue",

            f"${result['predicted_revenue_million']:,.2f}M"
        )

        st.subheader(
            "Top Drivers"
        )

        driver_df = pd.DataFrame(

            result[
                "top_drivers"
            ]
        )

        st.dataframe(
            driver_df,
            use_container_width=True
        )

        fig = px.bar(

            driver_df,

            x="feature",

            y="importance",

            title="Forecast Drivers"
        )

        st.plotly_chart(

            fig,

            use_container_width=True
        )


# ==========================================
# TAB 3
# ==========================================

with tab3:

    st.header(
        "Financial RAG Search"
    )

    query = st.text_input(

        "Search Financial Knowledge Base",

        "strong customer demand"
    )

    if st.button(
        "Search"
    ):

        response = requests.post(

            f"{API_URL}/rag/search",

            json={

                "query":
                query,

                "top_k":
                5
            }
        )

        result = response.json()

        for row in result[
            "results"
        ]:

            st.markdown("---")

            st.write(

                "Quarter:",

                row["quarter"]
            )

            st.write(

                "Distance:",

                round(
                    row["distance"],
                    4
                )
            )

            st.write(
                row["document"]
            )


# ==========================================
# TAB 4
# ==========================================

with tab4:

    st.header(
        "Feature Importance"
    )

    response = requests.get(

        f"{API_URL}/feature-importance"
    )

    result = response.json()

    importance_df = pd.DataFrame(

        result[
            "features"
        ]
    )

    st.dataframe(

        importance_df,

        use_container_width=True
    )

    fig = px.bar(

        importance_df,

        x="feature",

        y="importance",

        title="Feature Importance Ranking"
    )

    st.plotly_chart(

        fig,

        use_container_width=True
    )


# ==========================================
# DASHBOARD FOOTER
# ==========================================

st.markdown("---")

st.caption(

    "Financial Intelligence & Revenue Forecasting Copilot"
)