"""
Core plotting implementations.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._utils import (
    dispatch,
    process_matplotlib_kwargs,
    process_plotly_kwargs,
    process_vispy_kwargs,
)

if TYPE_CHECKING:
    from ._utils import ReturnType


@dispatch
def draw_mesh(
    vertices: np.ndarray | None = None, faces: np.ndarray | None = None, **kwargs
) -> ReturnType:
    """
    Plot a 3D mesh made of triangles or other polygon.

    Args:
        vertices: The array vertices.
        faces: The array of face indices.

    Returns:
        The resulting plot output.
    """


@draw_mesh.register("vispy")
def _(vertices, faces, **kwargs):
    from vispy.scene.visuals import Mesh

    canvas, view = process_vispy_kwargs(kwargs)

    view.add(Mesh(vertices, faces, shading="flat", **kwargs))
    view.camera.set_range()

    return canvas


@draw_mesh.register("matplotlib")
def _(vertices, faces, **kwargs):
    fig, ax = process_matplotlib_kwargs(kwargs)

    x, y, z = vertices.T
    i, j, k = faces.T

    return ax.plot_trisurf(x, y, z, triangles=faces, **kwargs)


@draw_mesh.register("plotly")
def _(vertices, faces, *args, **kwargs):
    fig = process_plotly_kwargs(kwargs)

    x, y, z = vertices.T
    i, j, k = faces.T

    return fig.add_mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, **kwargs)
