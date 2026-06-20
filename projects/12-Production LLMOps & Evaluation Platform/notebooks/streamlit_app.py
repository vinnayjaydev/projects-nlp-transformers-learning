import streamlit as st
import pandas as pd
import plotly.express as px

# ======================================================
# Page Configuration
# ======================================================

st.set_page_config(
    page_title="LLMOps Monitoring Platform",
    page_icon="📊",
    layout="wide"
)

# ======================================================
# Title
# ======================================================

st.title("📊 LLMOps Monitoring Dashboard")
st.markdown("Production Monitoring Platform")

# ======================================================
# Load Data
# ======================================================

try:

    telemetry_df = pd.read_csv(
        "../telemetry/trace_logs.csv"
    )

    drift_df = pd.read_csv(
        "../evaluation_results/drift_monitoring.csv"
    )

    faithfulness_df = pd.read_csv(
        "../evaluation_results/faithfulness_results.csv"
    )

    relevancy_df = pd.read_csv(
        "../evaluation_results/answer_relevancy_results.csv"
    )

    recall_df = pd.read_csv(
        "../evaluation_results/context_recall_results.csv"
    )

    judge_df = pd.read_csv(
        "../evaluation_results/llm_judge_results.csv"
    )

except Exception as e:

    st.error(f"Error loading files: {e}")
    st.stop()

# ======================================================
# Sidebar
# ======================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select View",
    [
        "Executive Dashboard",
        "Telemetry",
        "RAG Evaluation",
        "Drift Monitoring",
        "Hallucination Alerts",
        "Developer Diagnostics"
    ]
)

# ======================================================
# Executive Dashboard
# ======================================================

if page == "Executive Dashboard":

    st.header("Executive Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Requests",
        len(telemetry_df)
    )

    col2.metric(
        "Avg Latency",
        round(
            telemetry_df["latency_ms"].mean(),
            2
        )
    )

    col3.metric(
        "Total Cost",
        round(
            telemetry_df["cost_usd"].sum(),
            2
        )
    )

    col4.metric(
        "Avg Quality",
        round(
            telemetry_df["quality_score"].mean(),
            3
        )
    )

    fig = px.line(
        telemetry_df,
        y="latency_ms",
        title="Latency Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig = px.line(
        telemetry_df,
        y="quality_score",
        title="Quality Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ======================================================
# Telemetry
# ======================================================

elif page == "Telemetry":

    st.header("Live Telemetry")

    st.dataframe(
        telemetry_df,
        use_container_width=True
    )

    fig = px.histogram(
        telemetry_df,
        x="latency_ms",
        nbins=20,
        title="Latency Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig = px.histogram(
        telemetry_df,
        x="cost_usd",
        nbins=20,
        title="Cost Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ======================================================
# RAG Evaluation
# ======================================================

elif page == "RAG Evaluation":

    st.header("RAG Evaluation Metrics")

    metrics = {
        "Faithfulness":
            faithfulness_df["faithfulness"].mean(),

        "Relevancy":
            relevancy_df["answer_relevancy"].mean(),

        "Recall":
            recall_df["context_recall"].mean(),

        "Judge":
            judge_df["weighted_score"].mean() / 10
    }

    metric_df = pd.DataFrame(
        {
            "Metric": list(metrics.keys()),
            "Score": list(metrics.values())
        }
    )

    st.dataframe(
        metric_df,
        use_container_width=True
    )

    fig = px.bar(
        metric_df,
        x="Metric",
        y="Score",
        title="Average RAG Metrics"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ======================================================
# Drift Monitoring
# ======================================================

elif page == "Drift Monitoring":

    st.header("Semantic Drift Monitoring")

    fig = px.line(
        drift_df,
        x="month",
        y="distance",
        markers=True,
        title="Drift Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        drift_df,
        use_container_width=True
    )

    latest_drift = drift_df["distance"].iloc[-1]

    if latest_drift > 0.25:

        st.error(
            "🚨 Semantic Drift Detected"
        )

    else:

        st.success(
            "✅ No Significant Drift"
        )

# ======================================================
# Hallucination Alerts
# ======================================================

elif page == "Hallucination Alerts":

    st.header("Hallucination Monitoring")

    hallucinations = faithfulness_df[
        faithfulness_df["faithfulness"] < 0.70
    ]

    st.metric(
        "Hallucination Alerts",
        len(hallucinations)
    )

    st.dataframe(
        hallucinations,
        use_container_width=True
    )

# ======================================================
# Developer Diagnostics
# ======================================================

elif page == "Developer Diagnostics":

    st.header("Developer Diagnostics")

    diagnostic_df = pd.DataFrame(
        {
            "faithfulness":
                faithfulness_df["faithfulness"],

            "relevancy":
                relevancy_df["answer_relevancy"],

            "recall":
                recall_df["context_recall"]
        }
    )

    st.subheader("Correlation Matrix")

    st.dataframe(
        diagnostic_df.corr(),
        use_container_width=True
    )

    fig = px.scatter(
        diagnostic_df,
        x="faithfulness",
        y="relevancy",
        title="Faithfulness vs Relevancy"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )