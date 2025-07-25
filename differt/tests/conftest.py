from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import chex
import jax
import numpy as np
import pytest

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

if TYPE_CHECKING:
    from collections.abc import Iterator

    from jaxtyping import PRNGKeyArray
    from matplotlib.figure import Figure


def pytest_configure() -> None:
    chex.set_n_cpu_devices(8)


@pytest.fixture
def seed() -> int:
    return 1234


@pytest.fixture
def key(seed: int) -> PRNGKeyArray:
    return jax.random.key(seed)


@pytest.fixture
def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed=seed)


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


@pytest.fixture(autouse=True)
def close_figure(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[None]:
    if "backend" in request.fixturenames and plt is not None:
        figure_ = plt.figure
        fig = None

        @cache
        def figure() -> Figure:
            nonlocal fig
            fig = figure_()
            return fig

        with monkeypatch.context() as m:
            m.setattr(plt, "figure", figure)
            yield None

        if fig is not None:
            plt.close(fig)
    else:
        yield None
