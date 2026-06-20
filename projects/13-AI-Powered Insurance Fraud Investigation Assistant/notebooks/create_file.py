# create_notebooks.py

from pathlib import Path
import nbformat as nbf

notebooks = [
    "01-insurance-claims-corpus-builder.ipynb",
    "02-insurance-data-exploration.ipynb",
    "03-sentence-embedding-engine.ipynb",
    "04-cross-document-similarity-analysis.ipynb",
    "05-multi-statement-cross-examiner.ipynb",
    "06-template-fraud-detector.ipynb",
    "07-boilerplate-claim-clustering.ipynb",
    "08-cross-encoder-fraud-verification.ipynb",
    "09-collusion-network-construction.ipynb",
    "10-semantic-collusion-ring-detection.ipynb",
    "11-hybrid-lexical-semantic-investigation.ipynb",
    "12-evidence-synthesis-engine.ipynb",
    "13-llm-investigative-brief-generator.ipynb",
    "14-streamlit-siu-dashboard.ipynb",
    "15-end-to-end-insurance-fraud-assistant.ipynb",
]

current_folder = Path.cwd()

for notebook_name in notebooks:

    notebook = nbf.v4.new_notebook()

    title = notebook_name.replace(".ipynb", "")
    title = title.replace("-", " ").title()

    notebook.cells = [
        nbf.v4.new_markdown_cell(
            f"# {title}\n\n"
            f"**Project:** Insurance Fraud Detection Assistant\n\n"
            f"**Notebook:** `{notebook_name}`"
        ),
        nbf.v4.new_code_cell(
            "# Start coding here"
        )
    ]

    output_path = current_folder / notebook_name

    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(notebook, f)

    print(f"Created: {notebook_name}")

print("\nAll notebooks created successfully.")
print(f"Location: {current_folder}")
print(f"Location: {current_folder}")