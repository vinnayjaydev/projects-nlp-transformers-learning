from pathlib import Path
import nbformat as nbf

# List of notebook filenames to create
notebooks = [
    "01-building-a-personal-corpus.ipynb",
    "02-markdown-parser-and-chunking.ipynb",
    "03-semantic-embeddings.ipynb",
    "04-local-vector-store-faiss-chroma.ipynb",
    "05-hybrid-search-bm25-plus-sbert.ipynb",
    "06-metadata-filtering-and-time-aware-search.ipynb",
    "07-thought-graph-visualization.ipynb",
    "08-local-rag-over-personal-notes.ipynb",
    "09-streamlit-second-brain-app.ipynb",
    "10-end-to-end-personal-knowledge-engine.ipynb",
]

# Current working directory
output_dir = Path.cwd()

for notebook_name in notebooks:
    notebook_path = output_dir / notebook_name

    # Skip if the notebook already exists
    if notebook_path.exists():
        print(f"Already exists: {notebook_name}")
        continue

    # Create a new notebook
    nb = nbf.v4.new_notebook()

    # Add a default title markdown cell
    title = notebook_name.replace(".ipynb", "")
    title = title.split("-", 1)[1].replace("-", " ").title()

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            f"# {title}\n\n"
            f"**Notebook:** `{notebook_name}`\n\n"
            "This notebook is part of the Personal Knowledge Engine project."
        )
    )

    # Add an empty code cell
    nb.cells.append(
        nbf.v4.new_code_cell(
            "# Start coding here\n"
        )
    )

    # Write notebook to disk
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Created: {notebook_name}")

print("\nDone! All notebooks have been created in:")
print(output_dir)