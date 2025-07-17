__all__ = (
    "matplotlib",
    "plotly",
    "skip_if_matplotlib_not_installed",
    "skip_if_plotly_not_installed",
    "skip_if_vispy_not_installed",
    "vispy",
)


import importlib.util

import pytest

skip_if_vispy_not_installed = pytest.mark.skipif(
    importlib.util.find_spec("vispy") is None,
    reason="vispy not installed",
)
skip_if_matplotlib_not_installed = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="matplotlib not installed",
)
skip_if_plotly_not_installed = pytest.mark.skipif(
    importlib.util.find_spec("plotly") is None,
    reason="plotly not installed",
)

vispy = pytest.param(
    "vispy",
    marks=skip_if_vispy_not_installed,
    id="vispy",
)
matplotlib = pytest.param(
    "matplotlib",
    marks=skip_if_matplotlib_not_installed,
    id="matplotlib",
)
plotly = pytest.param(
    "plotly",
    marks=skip_if_plotly_not_installed,
    id="plotly",
)
