# ==========================================================
# Create Enterprise Multi-Document RAG Notebooks
# ==========================================================

from pathlib import Path
import nbformat as nbf

# ----------------------------------------------------------
# Notebook List
# ----------------------------------------------------------

notebooks = [
    "01-corporate-document-ingestion.ipynb",
    "02-layout-aware-document-parsing.ipynb",
    "03-parent-child-chunking.ipynb",
    "04-semantic-router.ipynb",
    "05-dual-vector-indexes.ipynb",
    "06-bm25-retrieval.ipynb",
    "07-semantic-vector-retrieval.ipynb",
    "08-hybrid-search-rrf-fusion.ipynb",
    "09-cross-encoder-reranking.ipynb",
    "10-citation-grounding-engine.ipynb",
    "11-conversational-memory.ipynb",
    "12-multi-document-rag-chain.ipynb",
    "13-rag-evaluation-ragas.ipynb",
    "14-fastapi-rag-api.ipynb",
    "15-streamlit-enterprise-assistant.ipynb",
    "16-production-system-design.ipynb",
]

# ----------------------------------------------------------
# Create Notebooks
# ----------------------------------------------------------

for notebook_name in notebooks:

    nb = nbf.v4.new_notebook()

    title = notebook_name.replace(".ipynb", "")
    title = title.replace("-", " ").title()

    nb.cells = [
        nbf.v4.new_markdown_cell(
            f"# {title}\n\n"
            f"**Project:** Enterprise Multi-Document RAG Assistant\n\n"
            f"**Notebook:** `{notebook_name}`"
        ),
        nbf.v4.new_code_cell(
            "# Start coding here\n"
        ),
    ]

    with open(notebook_name, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"✓ Created: {notebook_name}")

print("\nAll notebooks created successfully.")