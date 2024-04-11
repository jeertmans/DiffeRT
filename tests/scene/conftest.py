import pytest

from differt.scene.sionna import download_sionna_scenes


def pytest_sessionstart(session: pytest.Session) -> None:
    download_sionna_scenes()
