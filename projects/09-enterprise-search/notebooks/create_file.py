from pathlib import Path
import json

# ==========================================================
# Enterprise Search Notebook Files
# ==========================================================

NOTEBOOKS = [
    "01-building-enterprise-corpus.ipynb",
    "02-acl-and-metadata-indexing.ipynb",
    "03-bm25-lexical-search-engine.ipynb",
    "04-semantic-vector-search-faiss.ipynb",
    "05-hybrid-search-and-rrf-fusion.ipynb",
    "06-cross-encoder-reranking.ipynb",
    "07-access-control-and-secure-search.ipynb",
    "08-enterprise-search-api.ipynb",
    "09-streamlit-enterprise-search-dashboard.ipynb",
    "10-end-to-end-production-enterprise-search.ipynb",
]

# ==========================================================
# Minimal Jupyter Notebook Template
# ==========================================================

EMPTY_NOTEBOOK = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# ==========================================================
# Create Notebook Files
# ==========================================================

current_folder = Path.cwd()

print("=" * 60)
print("Creating Jupyter Notebook Files")
print("Current Folder:", current_folder)
print("=" * 60)

for notebook_name in NOTEBOOKS:

    notebook_path = current_folder / notebook_name

    if notebook_path.exists():
        print(f"[SKIPPED] {notebook_name} (already exists)")
        continue

    with open(
        notebook_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            EMPTY_NOTEBOOK,
            file,
            indent=2,
        )

    print(f"[CREATED] {notebook_name}")

print("=" * 60)
print(f"Completed! {len(NOTEBOOKS)} notebook files processed.")
print("=" * 60)