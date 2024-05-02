from pathlib import Path

import pytest

from differt.scene.sionna import download_sionna_scenes


def pytest_sessionstart(session: pytest.Session) -> None:
    download_sionna_scenes()


@pytest.fixture(scope="session")
def test_dir() -> Path:
    return Path(__file__).parent


@pytest.fixture(scope="session")
def project_dir(test_dir: Path) -> Path:
    return test_dir.parent


@pytest.fixture(scope="session")
def pyproject_toml(project_dir: Path) -> Path:
    return project_dir.joinpath("pyproject.toml").resolve(strict=True)


@pytest.fixture(scope="session")
def cargo_toml(project_dir: Path) -> Path:
    return project_dir.joinpath("Cargo.toml").resolve(strict=True)
