# ==========================================================
# AI Resume Search Dashboard
# streamlit_app.py
# ==========================================================

import requests
import pandas as pd
import streamlit as st

# ==========================================================
# Configuration
# ==========================================================

API_URL = "http://127.0.0.1:8000/match"

st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="📄",
    layout="wide",
)

# ==========================================================
# Header
# ==========================================================

st.title("🤖 AI Resume Matcher")
st.markdown(
    """
Semantic Resume Screening using **SBERT + FAISS + FastAPI + Streamlit**

This dashboard sends the Job Description to the FastAPI backend,
which performs semantic retrieval and returns the top matching resumes.
"""
)

st.divider()

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.header("⚙️ Search Settings")

top_k = st.sidebar.slider(
    label="Top Candidates",
    min_value=1,
    max_value=10,
    value=5,
)

st.sidebar.info(
    """
1. Enter the Job Description.
2. Click **Match Candidates**.
3. View the ranked resumes.
4. Download the results as CSV.
"""
)

# ==========================================================
# Job Description Input
# ==========================================================

default_jd = """Need an accountant with:
- Financial Accounting
- GST and TDS
- Tally ERP
- SAP FICO
- Advanced Excel
- Minimum 3 years of experience
"""

st.subheader("📋 Job Description")

job_description = st.text_area(
    label="Paste the Job Description",
    value=default_jd,
    height=220,
)

# ==========================================================
# Match Button
# ==========================================================

if st.button("🔍 Match Candidates", use_container_width=True):

    if job_description.strip() == "":
        st.warning("Please enter a Job Description.")
        st.stop()

    payload = {
        "job_description": job_description,
        "top_k": top_k,
    }

    try:

        with st.spinner("Searching for matching resumes..."):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=60,
            )

        if response.status_code != 200:
            st.error(
                f"API Error ({response.status_code})"
            )
            st.code(response.text)
            st.stop()

        results = response.json()

        # ==================================================
        # Display Summary
        # ==================================================

        st.success("Search completed successfully!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Candidates Returned",
                value=results["num_results"],
            )

        with col2:
            st.metric(
                label="Top K Requested",
                value=top_k,
            )

        st.divider()

        # ==================================================
        # Candidate Leaderboard
        # ==================================================

        st.subheader("🏆 Candidate Rankings")

        leaderboard = pd.DataFrame(
            results["results"]
        )

        leaderboard.index = leaderboard.index + 1

        st.dataframe(
            leaderboard,
            use_container_width=True,
        )

        # ==================================================
        # Candidate Details
        # ==================================================

        st.subheader("📄 Candidate Details")

        for row in results["results"]:

            with st.expander(
                f"📄 {row['candidate']}"
            ):

                st.write(
                    f"**Semantic Match Score:** "
                    f"`{row['faiss_score']}`"
                )

                if "cross_score" in row:
                    st.write(
                        f"**Cross Encoder Score:** "
                        f"`{row['cross_score']:.4f}`"
                    )

                if "matched_skills" in row:
                    st.success(
                        "Matched Skills: "
                        + ", ".join(
                            row["matched_skills"]
                        )
                    )

                if "missing_skills" in row:
                    st.warning(
                        "Missing Skills: "
                        + ", ".join(
                            row["missing_skills"]
                        )
                    )

        # ==================================================
        # Download CSV
        # ==================================================

        csv = leaderboard.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇ Download Results CSV",
            data=csv,
            file_name="resume_matches.csv",
            mime="text/csv",
        )

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI backend.\n\n"
            "Make sure your API is running:\n"
            "http://127.0.0.1:8000"
        )

    except Exception as e:
        st.exception(e)

# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    "AI Resume Matcher • "
    "SBERT + FAISS + FastAPI + Streamlit"
)