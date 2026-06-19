# ==========================================================
# Create LangGraph Multi-Agent Research Assistant Notebooks
# ==========================================================

from pathlib import Path
import nbformat as nbf

notebooks = [
    "01-langgraph-fundamentals.ipynb",
    "02-state-management.ipynb",
    "03-researcher-agent.ipynb",
    "04-semantic-deduplication-agent.ipynb",
    "05-critic-agent-and-self-correction.ipynb",
    "06-semantic-gap-analysis.ipynb",
    "07-fact-checker-agent.ipynb",
    "08-human-in-the-loop.ipynb",
    "09-parallel-agent-execution.ipynb",
    "10-langgraph-compilation.ipynb",
    "11-enterprise-research-report-generator.ipynb",
    "12-streamlit-mission-control-dashboard.ipynb",
    "13-end-to-end-multi-agent-system.ipynb",
]

current_folder = Path.cwd()

for notebook_name in notebooks:

    notebook = nbf.v4.new_notebook()

    notebook.cells = [
        nbf.v4.new_markdown_cell(
            f"# {notebook_name.replace('.ipynb', '').replace('-', ' ').title()}"
        ),
        nbf.v4.new_code_cell(
            "# Start coding here"
        ),
    ]

    notebook_path = current_folder / notebook_name

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(notebook, f)

    print(f"Created: {notebook_name}")

print("\nAll notebooks created successfully.")
print(f"Location: {current_folder}")