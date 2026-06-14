# ==========================================================
# Enterprise Search Dashboard
# Streamlit + FastAPI + Hybrid Search
# ==========================================================

import requests
import pandas as pd
import streamlit as st

# ==========================================================
# Configuration
# ==========================================================

API_URL = "http://127.0.0.1:8000/search"

st.set_page_config(
    page_title="Enterprise Search Dashboard",
    page_icon="🔍",
    layout="wide",
)

# ==========================================================
# Header
# ==========================================================

st.title("🔍 Enterprise Search Dashboard")

st.markdown(
    """
Secure Hybrid Enterprise Search powered by:

- 🔎 BM25 Lexical Search
- 🧠 SBERT + FAISS Semantic Search
- 🔀 Hybrid Retrieval
- 🎯 Cross-Encoder Re-ranking
- 🔒 ACL-based Access Control
"""
)

st.divider()

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.header("👤 User Context")

selected_user = st.sidebar.selectbox(
    "Select User",
    [
        "alice",
        "bob",
        "carol",
        "david",
    ],
)

top_k = st.sidebar.slider(
    "Top K Results",
    min_value=1,
    max_value=10,
    value=5,
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
**Demo Users**

- alice → Engineering
- bob → HR
- carol → Finance
- david → Legal
"""
)

# ==========================================================
# Search Input
# ==========================================================

st.subheader("🔎 Enterprise Search")

query = st.text_input(
    "Search Query",
    placeholder="Example: vector database migration",
)

search_button = st.button(
    "🔍 Search",
    use_container_width=True,
)

# ==========================================================
# Search Logic
# ==========================================================

if search_button:

    if query.strip() == "":

        st.warning(
            "Please enter a search query."
        )

    else:

        payload = {
            "user_name": selected_user,
            "query": query,
            "top_k": top_k,
        }

        try:

            response = requests.post(
                API_URL,
                json=payload,
                timeout=60,
            )

            if response.status_code == 200:

                data = response.json()

                results = pd.DataFrame(
                    data["results"]
                )

                st.success(
                    f"Found {len(results)} matching documents."
                )

                # ==================================================
                # Search Statistics
                # ==================================================

                st.subheader("📈 Search Statistics")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Results Returned",
                    len(results),
                )

                col2.metric(
                    "User",
                    selected_user,
                )

                if len(results) > 0 and "score" in results.columns:
                    top_score = results["score"].max()
                else:
                    top_score = 0.0

                col3.metric(
                    "Top Score",
                    f"{top_score:.2f}",
                )

                st.divider()

                # ==================================================
                # Results Table
                # ==================================================

                st.subheader("📋 Search Results")

                preferred_columns = [
                    "doc_id",
                    "title",
                    "department",
                    "score",
                ]

                display_columns = [
                    col
                    for col in preferred_columns
                    if col in results.columns
                ]

                st.dataframe(
                    results[
                        display_columns
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # ==================================================
                # Document Explorer
                # ==================================================

                st.subheader("📄 Document Explorer")

                for _, row in results.iterrows():

                    title = row.get(
                        "title",
                        "Untitled Document",
                    )

                    score = row.get(
                        "score",
                        0.0,
                    )

                    with st.expander(
                        f"{title}   |   Score: {score:.2f}"
                    ):

                        if "doc_id" in row:
                            st.markdown(
                                f"**Document ID:** `{row['doc_id']}`"
                            )

                        if "department" in row:
                            st.markdown(
                                f"**Department:** {row['department']}"
                            )

                        if "acl_roles" in row:
                            st.markdown(
                                f"**ACL Roles:** {row['acl_roles']}"
                            )

                        if "source" in row:
                            st.markdown(
                                f"**Source:** {row['source']}"
                            )

                        if "content" in row:
                            st.markdown(
                                "**Content Preview**"
                            )

                            st.info(
                                str(
                                    row["content"]
                                )[:500]
                                + "..."
                            )

            else:

                st.error(
                    f"API Error ({response.status_code})"
                )

                st.code(
                    response.text,
                    language="text",
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the Enterprise Search API."
            )

            st.info(
                """
Please make sure your FastAPI server is running.

Example:

uvicorn app:app --reload
"""
            )

        except Exception as e:

            st.error(
                f"Unexpected error: {str(e)}"
            )

# ==========================================================
# Footer
# ==========================================================

st.markdown("---")

st.caption(
    "Enterprise Search Dashboard • Hybrid Search • FAISS • BM25 • Cross Encoder • FastAPI • Streamlit"
)