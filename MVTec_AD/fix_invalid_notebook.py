from pathlib import Path

import nbformat

path = Path("./vit_knn.ipynb")

# Load the notebook
with open(path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Check global metadata
if "widgets" in nb.metadata:
    if "state" not in nb.metadata["widgets"]:
        print("Removing corrupted notebook-level 'widgets' metadata (missing 'state').")
        del nb.metadata["widgets"]

# Also remove cell-level widgets metadata if missing 'state'
for i, cell in enumerate(nb.cells):
    if "widgets" in cell.get("metadata", {}):
        if "state" not in cell["metadata"]["widgets"]:
            print(f"Removing cell {i} corrupted 'widgets' metadata.")
            del cell["metadata"]["widgets"]

# Save the cleaned notebook
with open(path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(" Cleaned notebook saved successfully.")
