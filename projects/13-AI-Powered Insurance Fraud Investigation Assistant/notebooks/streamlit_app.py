# ==========================================
# Insurance Fraud Investigation Dashboard
# Streamlit SIU Dashboard
# ==========================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================
# Page Config
# ==========================================

st.set_page_config(

    page_title="Insurance Fraud SIU Dashboard",

    page_icon="🚨",

    layout="wide"
)

# ==========================================
# Data Path
# ==========================================

DATA_DIR = r"D:\Books\projects-nlp-transformers-learning\projects\13-AI-Powered Insurance Fraud Investigation Assistant\data"

# ==========================================
# Load Data
# ==========================================

@st.cache_data
def load_data():

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

    briefs_df = pd.read_csv(

        os.path.join(
            DATA_DIR,
            "llm_investigation_briefs.csv"
        )
    )

    return claims_df, risk_df, briefs_df


claims_df, risk_df, briefs_df = load_data()

# ==========================================
# Sidebar
# ==========================================

st.sidebar.title(
    "🚨 SIU Command Center"
)

page = st.sidebar.radio(

    "Navigation",

    [

        "Dashboard",

        "Investigation Queue",

        "Claim Search",

        "Risk Analytics",

        "Investigation Briefs"

    ]
)

# ==========================================
# Dashboard
# ==========================================

if page == "Dashboard":

    st.title(
        "🚨 Insurance Fraud Investigation Dashboard"
    )

    total_claims = len(claims_df)

    high_risk = len(

        risk_df[
            risk_df[
                "fraud_risk_index"
            ] >= 60
        ]
    )

    critical_risk = len(

        risk_df[
            risk_df[
                "fraud_risk_index"
            ] >= 80
        ]
    )

    avg_risk = round(

        risk_df[
            "fraud_risk_index"
        ].mean(),

        2
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Claims",
        total_claims
    )

    col2.metric(
        "High Risk Claims",
        high_risk
    )

    col3.metric(
        "Critical Claims",
        critical_risk
    )

    col4.metric(
        "Average Risk",
        avg_risk
    )

    st.markdown("---")

    fig = px.histogram(

        risk_df,

        x="fraud_risk_index",

        nbins=20,

        title="Fraud Risk Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================
# Investigation Queue
# ==========================================

elif page == "Investigation Queue":

    st.title(
        "🕵 Investigation Queue"
    )

    queue_df = risk_df.sort_values(

        by="fraud_risk_index",

        ascending=False
    )

    st.dataframe(

        queue_df,

        use_container_width=True
    )

    st.download_button(

        label="Download Queue",

        data=queue_df.to_csv(
            index=False
        ),

        file_name="investigation_queue.csv",

        mime="text/csv"
    )

# ==========================================
# Claim Search
# ==========================================

elif page == "Claim Search":

    st.title(
        "🔍 Claim Search"
    )

    claim_ids = claims_df[
        "claim_id"
    ].tolist()

    selected_claim = st.selectbox(

        "Select Claim",

        claim_ids
    )

    claim = claims_df[

        claims_df[
            "claim_id"
        ] == selected_claim

    ].iloc[0]

    risk = risk_df[

        risk_df[
            "claim_id"
        ] == selected_claim

    ]

    st.subheader(
        "Claim Information"
    )

    st.write(
        claim.to_dict()
    )

    if len(risk) > 0:

        st.subheader(
            "Fraud Risk"
        )

        st.dataframe(
            risk,
            use_container_width=True
        )

# ==========================================
# Risk Analytics
# ==========================================

elif page == "Risk Analytics":

    st.title(
        "📊 Fraud Risk Analytics"
    )

    st.subheader(
        "Risk Category Distribution"
    )

    category_counts = risk_df[
        "risk_category"
    ].value_counts()

    fig = px.bar(

        x=category_counts.index,

        y=category_counts.values,

        labels={
            "x":"Risk Category",
            "y":"Count"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Top High Risk Claims"
    )

    top_risk = risk_df.sort_values(

        by="fraud_risk_index",

        ascending=False

    ).head(20)

    st.dataframe(
        top_risk,
        use_container_width=True
    )

# ==========================================
# Investigation Briefs
# ==========================================

elif page == "Investigation Briefs":

    st.title(
        "📄 Investigation Briefs"
    )

    claim_ids = briefs_df[
        "claim_id"
    ].tolist()

    selected_claim = st.selectbox(

        "Select Claim ID",

        claim_ids
    )

    brief = briefs_df[

        briefs_df[
            "claim_id"
        ] == selected_claim

    ]

    if len(brief) > 0:

        st.subheader(
            f"Investigation Brief - {selected_claim}"
        )

        st.text_area(

            "Generated Investigation Brief",

            value=str(

                brief.iloc[0][
                    "investigation_brief"
                ]
            ),

            height=400
        )

# ==========================================
# Footer
# ==========================================

st.markdown("---")

st.caption(
    "Insurance Fraud Investigation & Audit Assistant"
)