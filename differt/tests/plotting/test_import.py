import importlib

import pytest

import differt.plotting._core

from ._types import MissingModulesContextGenerator


@pytest.mark.parametrize(
    "backends",
    (("vispy",), ("matplotlib",), ("plotly",), ("vispy", "matplotlib", "plotly")),
)
def test_import_with_missing_backends(
    backends: tuple[str],
    missing_modules: MissingModulesContextGenerator,
) -> None:
    with missing_modules(*backends):
        importlib.reload(differt.plotting._core)
