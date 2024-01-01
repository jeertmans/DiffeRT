"""
Plotting utils for DiffeRT objects.
"""
__all__ = (
    "dispatch",
    "use",
    "draw_mesh",
)

from ._core import draw_mesh
from ._decorators import dispatch
