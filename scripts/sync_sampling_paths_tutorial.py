"""Download and synchronize the sampling paths tutorial notebook from remote repository."""

import sys
from pathlib import Path

import requests

if __name__ == "__main__":
    response = requests.get(
        "https://raw.githubusercontent.com/jeertmans/sampling-paths/main/notebooks/tutorial.ipynb",
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
        existing_content = Path(target_file).read_bytes()
        if existing_content != response.content:
            file_changed = True
    if file_changed:
        Path(target_file).write_bytes(response.content)
    if file_changed:
        sys.exit(1)
