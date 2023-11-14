from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_dir() -> Path:
    return Path(__file__).parent


@pytest.fixture(scope="session")
def project_dir(test_dir: Path) -> Path:
    return test_dir.parent


@pytest.fixture(scope="session")
def differt_core_dir(project_dir: Path) -> Path:
    return project_dir.joinpath("differt-core")


@pytest.fixture(scope="session")
def differt_pyproject_toml(project_dir: Path) -> Path:
    return project_dir.joinpath("pyproject.toml").resolve(strict=True)


@pytest.fixture(scope="session")
def differt_core_pyproject_toml(differt_core_dir: Path) -> Path:
    return differt_core_dir.joinpath("pyproject.toml").resolve(strict=True)


@pytest.fixture(scope="session")
def differt_core_cargo_toml(differt_core_dir: Path) -> Path:
    return differt_core_dir.joinpath("Cargo.toml").resolve(strict=True)
