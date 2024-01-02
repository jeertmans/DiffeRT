"""
Plotting utilities for DiffeRT objects.
"""
__all__ = (
    "dispatch",
    "use",
    "draw_mesh",
    "process_vispy_kwargs",
    "process_matplotlib_kwargs",
    "process_plotly_kwargs",
    "view_from_canvas",
)

from ._core import draw_mesh
from ._utils import (
    dispatch,
    process_matplotlib_kwargs,
    process_plotly_kwargs,
    process_vispy_kwargs,
    use,
    view_from_canvas,
)
