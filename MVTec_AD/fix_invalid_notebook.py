# this is a script to fix if your jupyter notebook has corrupted widgets metadata
from pathlib import Path

import nbformat

path = Path("enter your notebook path here")

with open(path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

if "widgets" in nb.metadata:
    if "state" not in nb.metadata["widgets"]:
        print("Removing corrupted notebook-level 'widgets' metadata (missing 'state').")
        del nb.metadata["widgets"]

for i, cell in enumerate(nb.cells):
    if "widgets" in cell.get("metadata", {}):
        if "state" not in cell["metadata"]["widgets"]:
            print(f"Removing cell {i} corrupted 'widgets' metadata.")
            del cell["metadata"]["widgets"]

with open(path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(" Cleaned notebook saved successfully.")
