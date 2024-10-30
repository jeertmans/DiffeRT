"""Read a list of Jupyter Notebook files and append a installation preamble if needed."""

import sys

import nbformat as nbf

TAG = "differt-install-preamble"

CELL = nbf.NotebookNode(
    cell_type="code",
    execution_count=None,
    id="0",
    metadata={"tags": [TAG, "remove-cell", "skip-execution"]},
    outputs=[],
    source=[
        "# Run this cell to install DiffeRT and its dependencies, e.g., on Google Colab\n",
        "\n",
        "try:\n",
        "    import differt  # noqa: F401\n",
        "except ImportError:\n",
        "    import sys  # noqa: F401\n",
        "\n",
        "    !{sys.executable} -m pip install differt[all]",
    ],
)

if __name__ == "__main__":
    for ipath in sys.argv[1:]:
        ntbk = nbf.read(ipath, nbf.NO_CONVERT)

        cells = ntbk.cells

        if len(cells) > 1 and TAG in cells[0].get("metadata", {}).get("tags", []):
            cell = cells[0]

            if "nbsetup-skip" in cell["metadata"]["tags"]:
                continue  # Skip this cell, we don't to overwrite it

            for field, value in CELL.items():
                cell[field] = value
        else:
            cells.insert(0, CELL)

        nbf.write(ntbk, ipath)
