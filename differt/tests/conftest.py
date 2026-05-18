from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import numpy as np
import pytest

try:
    import matplotlib.pyplot  # noqa: ICN001
except ImportError:
    plt = None
else:
    plt = matplotlib.pyplot

if TYPE_CHECKING:
    from collections.abc import Iterator

    from jaxtyping import PRNGKeyArray
    from matplotlib.figure import Figure


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


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "require_typechecker: mark test as requiring runtime type checking"
    )
    config.addinivalue_line(
        "markers",
        "require_no_typechecker: mark test as requiring no runtime type checking",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if (option := config.getoption("--jaxtyping-packages")) and "differt" in option:
        skip_require_no_typechecker = pytest.mark.skip(
            reason='cannot run with --jaxtyping-packages="differt,..." option enabled'
        )
        for item in items:
            if "require_no_typechecker" in item.keywords:
                item.add_marker(skip_require_no_typechecker)
        return
    skip_require_typechecker = pytest.mark.skip(
        reason='need --jaxtyping-packages="differt,..." option to run'
    )
    for item in items:
        if "require_typechecker" in item.keywords:
            item.add_marker(skip_require_typechecker)
