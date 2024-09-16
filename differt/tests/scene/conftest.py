from pathlib import Path

import pytest

from differt.scene.sionna import SIONNA_SCENES_FOLDER, download_sionna_scenes


@pytest.fixture(scope="session")
def sionna_folder() -> Path:
    download_sionna_scenes(folder=SIONNA_SCENES_FOLDER)
    return SIONNA_SCENES_FOLDER
