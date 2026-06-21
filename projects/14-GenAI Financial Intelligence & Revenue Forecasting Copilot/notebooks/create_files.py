# create_notebooks.py

from pathlib import Path
import nbformat as nbf

notebooks = [
    "01-financial-data-sandbox.ipynb",
    "02-financial-document-ingestion.ipynb",
    "03-financial-text-exploration.ipynb",
    "04-finbert-sentiment-analysis.ipynb",
    "05-forward-looking-statement-detector.ipynb",
    "06-risk-factor-semantic-analysis.ipynb",
    "07-financial-embedding-engine.ipynb",
    "08-macroeconomic-similarity-engine.ipynb",
    "09-financial-rag-audit-engine.ipynb",
    "10-baseline-revenue-forecasting.ipynb",
    "11-multimodal-feature-fusion.ipynb",
    "12-multimodal-revenue-forecasting.ipynb",
    "13-forecast-rationale-generator.ipynb",
    "14-financial-intelligence-api.ipynb",
    "15-fpa-command-center-dashboard.ipynb",
]

current_folder = Path.cwd()

for notebook_name in notebooks:

    nb = nbf.v4.new_notebook()

    title = notebook_name.replace(".ipynb", "")
    title = title.split("-", 1)[1].replace("-", " ").title()

    markdown_cell = nbf.v4.new_markdown_cell(
        f"# {title}\n\n"
        f"**Project:** Financial Planning & Analysis Intelligence Platform\n\n"
        f"**Notebook:** `{notebook_name}`"
    )

    code_cell = nbf.v4.new_code_cell(
        "# Start coding here\n"
    )

    nb["cells"] = [
        markdown_cell,
        code_cell,
    ]

    output_path = current_folder / notebook_name

    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Created: {notebook_name}")

print("\nAll notebooks created successfully.")
print(f"Location: {current_folder}")