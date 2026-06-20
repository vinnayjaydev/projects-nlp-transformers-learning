from pathlib import Path
import nbformat as nbf

notebooks = [
    "01-golden-dataset-assembly.ipynb",
    "02-rouge-and-heuristic-metrics.ipynb",
    "03-bertscore-semantic-evaluation.ipynb",
    "04-sentence-embedding-similarity.ipynb",
    "05-unified-evaluation-engine.ipynb",
    "06-faithfulness-evaluator.ipynb",
    "07-answer-relevancy-evaluator.ipynb",
    "08-context-recall-evaluator.ipynb",
    "09-llm-as-a-judge-engine.ipynb",
    "10-rag-evaluation-platform.ipynb",
    "11-semantic-drift-detection.ipynb",
    "12-telemetry-and-trace-logging.ipynb",
    "13-llmops-fastapi-monitoring-api.ipynb",
    "14-streamlit-llmops-dashboard.ipynb",
    "15-end-to-end-llmops-platform.ipynb",
]

for notebook_name in notebooks:

    nb = nbf.v4.new_notebook()

    title = notebook_name.replace(".ipynb", "")
    title = title.replace("-", " ").title()

    nb.cells = [
        nbf.v4.new_markdown_cell(
            f"# {title}\n\n"
            f"**Project:** LLMOps Monitoring & Evaluation Platform\n\n"
            f"**Notebook:** `{notebook_name}`"
        ),
        nbf.v4.new_code_cell(
            "# Start coding here\n"
        ),
    ]

    with open(notebook_name, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

print(f"Created {len(notebooks)} notebooks successfully.")