#!/usr/bin/env uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "nbdime>=4.0.4",
#     "nbformat>=5.11",
#     "requests>=2.34.2",
# ]
# ///
"""Download and synchronize the sampling paths tutorial notebook from remote repository."""

import sys
from pathlib import Path

import nbformat
import requests
from nbdime import diff_notebooks  # ty: ignore[unresolved-import]
from nbdime.prettyprint import (  # ty: ignore[unresolved-import]
    PrettyPrintConfig,
    pretty_print_notebook_diff,
)

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/jeertmans/sampling-paths/main/notebooks/tutorial.ipynb"
    response = requests.get(
        url,
        timeout=60,
    )
    response.raise_for_status()

    target_file = (
        Path(__file__).parent.parent
        / "docs"
        / "source"
        / "notebooks"
        / "sampling_paths.ipynb"
    )
    file_changed = False

    if target_file.exists():
        existing_content = target_file.read_bytes()
        if existing_content != response.content:
            base = nbformat.reads(existing_content.decode("utf-8"), as_version=4)
            remote = nbformat.reads(response.content.decode("utf-8"), as_version=4)

            diff = diff_notebooks(base, remote)
            config = PrettyPrintConfig(out=sys.stdout)
            pretty_print_notebook_diff(str(target_file), "<remote>", base, diff, config)

            target_file.write_bytes(response.content)
            sys.exit(1)
