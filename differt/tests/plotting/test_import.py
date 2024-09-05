import importlib

import pytest
from pytest_missing_modules.plugin import MissingModulesContextGenerator

import differt.plotting._core


@pytest.mark.parametrize(
    "backends",
    [("vispy",), ("matplotlib",), ("plotly",), ("vispy", "matplotlib", "plotly")],
)
def test_import_with_missing_backends(
    backends: tuple[str],
    missing_modules: MissingModulesContextGenerator,
) -> None:
    with missing_modules(*backends):
        importlib.reload(differt.plotting._core)
