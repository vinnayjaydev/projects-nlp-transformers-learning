from pathlib import Path
import json

# ==========================================================
# Configuration
# ==========================================================

NOTEBOOK_FOLDER = Path(".")

NOTEBOOKS = [
    "01-pdf-parsing.ipynb",
    "02-recursive-chunking.ipynb",
    "03-embedding-generation.ipynb",
    "04-faiss-vector-index.ipynb",
    "05-grounded-rag-pipeline.ipynb",
    "06-conversational-memory.ipynb",
    "07-rag-evaluation-and-debugging.ipynb",
    "08-streamlit-pdf-chatbot.ipynb",
    "09-end-to-end-pdf-chatbot.ipynb",
    "10-production-rag-system-design.ipynb",
]

# ==========================================================
# Create Folder
# ==========================================================

NOTEBOOK_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

# ==========================================================
# Base Notebook Template
# ==========================================================

def create_notebook_content(title):

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n",
                    "\n",
                    "This notebook is part of the **Production-Style PDF Chatbot / RAG System Project**.\n",
                    "\n",
                    "## Objectives\n",
                    "- Understand the underlying concepts.\n",
                    "- Build a production-quality implementation.\n",
                    "- Learn the theory behind every code block.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Start coding here\n"
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    return notebook


# ==========================================================
# Create Notebooks
# ==========================================================

for notebook_name in NOTEBOOKS:

    notebook_path = NOTEBOOK_FOLDER / notebook_name

    if notebook_path.exists():
        print(f"Already exists: {notebook_name}")
        continue

    title = (
        notebook_name
        .replace(".ipynb", "")
        .replace("-", " ")
        .title()
    )

    notebook_data = create_notebook_content(title)

    with open(
        notebook_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            notebook_data,
            f,
            indent=2,
        )

    print(f"Created: {notebook_name}")

print("\n" + "=" * 60)
print("All notebook files have been created successfully!")
print("Location:", NOTEBOOK_FOLDER.resolve())
print("=" * 60)