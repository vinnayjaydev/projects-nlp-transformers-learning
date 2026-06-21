# ==========================================
# Hybrid Recommendation Dashboard
# streamlit_app.py
# ==========================================

import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Recommendation Copilot",
    page_icon="🎯",
    layout="wide"
)

# ==========================================
# LOAD DATA
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

interactions_df = pd.read_csv(
    "../data/user_interactions.csv"
)

# ==========================================
# TITLE
# ==========================================

st.title(
    "🎯 Hybrid LLM Recommendation System"
)

st.markdown(
    """
    Personalized Recommendation Analytics Dashboard
    """
)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header(
    "User Controls"
)

user_ids = sorted(
    recommendations_df["user_id"].unique()
)

selected_user = st.sidebar.selectbox(
    "Select User",
    user_ids
)

# ==========================================
# USER HISTORY
# ==========================================

st.header(
    "📚 User History"
)

user_history = interactions_df[
    interactions_df["user_id"]
    == selected_user
]

history_df = pd.merge(
    user_history,
    profiles_df[
        [
            "item_id",
            "title",
            "category"
        ]
    ],
    on="item_id",
    how="left"
)

st.dataframe(
    history_df,
    use_container_width=True
)

# ==========================================
# RECOMMENDATIONS
# ==========================================

st.header(
    "⭐ Personalized Recommendations"
)

user_recommendations = recommendations_df[
    recommendations_df["user_id"]
    == selected_user
]

st.dataframe(
    user_recommendations,
    use_container_width=True
)

# ==========================================
# RECOMMENDATION BAR CHART
# ==========================================

if "hybrid_score" in user_recommendations.columns:

    fig = px.bar(
        user_recommendations,
        x="title",
        y="hybrid_score",
        title="Recommendation Scores"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================
# ITEM CATALOG
# ==========================================

st.header(
    "📦 Item Catalog"
)

st.dataframe(
    profiles_df,
    use_container_width=True
)

# ==========================================
# CATEGORY DISTRIBUTION
# ==========================================

st.header(
    "📊 Category Distribution"
)

category_counts = profiles_df[
    "category"
].value_counts()

fig = px.pie(
    values=category_counts.values,
    names=category_counts.index,
    title="Catalog Categories"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# USER EVALUATION
# ==========================================

st.header(
    "📈 User Evaluation Metrics"
)

user_metrics = evaluation_df[
    evaluation_df["user_id"]
    == selected_user
]

st.dataframe(
    user_metrics,
    use_container_width=True
)

# ==========================================
# SUMMARY METRICS
# ==========================================

st.header(
    "🚀 Recommendation System Metrics"
)

st.dataframe(
    summary_df,
    use_container_width=True
)

# ==========================================
# METRICS VISUALIZATION
# ==========================================

fig = px.bar(
    summary_df,
    x="metric",
    y="value",
    title="System Evaluation Metrics"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# TOP RECOMMENDED ITEMS
# ==========================================

st.header(
    "🏆 Most Recommended Items"
)

top_items = recommendations_df[
    "title"
].value_counts().reset_index()

top_items.columns = [
    "title",
    "count"
]

fig = px.bar(
    top_items.head(10),
    x="title",
    y="count",
    title="Top Recommended Items"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# COVERAGE ANALYSIS
# ==========================================

st.header(
    "🌎 Recommendation Coverage"
)

recommended_items = recommendations_df[
    "item_id"
].nunique()

catalog_items = profiles_df[
    "item_id"
].nunique()

coverage = (
    recommended_items
    /
    catalog_items
)

st.metric(
    "Coverage",
    round(
        coverage,
        3
    )
)

# ==========================================
# RAW DATA
# ==========================================

with st.expander(
    "View Raw Recommendation Data"
):

    st.dataframe(
        recommendations_df,
        use_container_width=True
    )

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    """
    Hybrid Recommendation System

    Collaborative Filtering +
    Semantic Search +
    Hybrid Fusion +
    LLM Re-Ranking
    """
)