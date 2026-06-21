from pathlib import Path

notebooks = [
    "01-recommendation-data-sandbox.ipynb",
    "02-recommendation-data-exploration.ipynb",
    "03-content-profiling-engine.ipynb",
    "04-item-embedding-generator.ipynb",
    "05-vector-index-construction.ipynb",
    "06-semantic-item-similarity-engine.ipynb",
    "07-user-profile-vector-engine.ipynb",
    "08-personalized-candidate-retrieval.ipynb",
    "09-cold-start-recommendation-engine.ipynb",
    "10-maximal-marginal-relevance-mmr.ipynb",
    "11-hybrid-recommendation-engine.ipynb",
    "12-llm-reranker-and-explainer.ipynb",
    "13-recommendation-evaluation-framework.ipynb",
    "14-recommendation-analytics-dashboard.ipynb",
    "15-end-to-end-personalized-recommendation-system.ipynb",
]

notebook_template = """{
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
"""

for notebook in notebooks:
    path = Path(notebook)

    if not path.exists():
        path.write_text(notebook_template, encoding="utf-8")
        print(f"Created: {notebook}")
    else:
        print(f"Already exists: {notebook}")

print(f"\nTotal notebooks processed: {len(notebooks)}")