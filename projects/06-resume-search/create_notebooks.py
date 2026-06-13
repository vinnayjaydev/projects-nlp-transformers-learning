import os
import json

# List of notebook files
notebook_files = [
    "01-document-parsing.ipynb",
    "02-section-extraction.ipynb",
    "03-resume-embeddings.ipynb",
    "04-section-weighted-matching.ipynb",
    "05-skill-gap-analysis.ipynb",
    "06-faiss-candidate-retrieval.ipynb",
    "07-cross-encoder-reranking.ipynb",
    "08-bias-mitigation.ipynb",
    "09-fastapi-resume-api.ipynb",
    "10-streamlit-dashboard.ipynb",
    "11-end-to-end-ai-ats-system.ipynb",
]

# Create notebooks directory in the current folder
notebooks_dir = os.path.join(os.getcwd(), "notebooks")
os.makedirs(notebooks_dir, exist_ok=True)

# Minimal valid Jupyter notebook template
notebook_template = {
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

# Create each notebook file
for notebook_name in notebook_files:
    notebook_path = os.path.join(notebooks_dir, notebook_name)

    # Only create if it doesn't already exist
    if not os.path.exists(notebook_path):
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook_template, f, indent=2)
        print(f"Created: {notebook_path}")
    else:
        print(f"Already exists: {notebook_path}")

print("\n✅ All notebook files have been created successfully!")