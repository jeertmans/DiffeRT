"""
Core plotting implementations.
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from ._decorators import dispatch

if TYPE_CHECKING:
    from plotly.graph_objects import Figure
    from vispy.app.canvas import Canvas


@dispatch
def draw_mesh(vertices: np.ndarray | None = None, faces: np.ndarray | None = None, *args, **kwargs) -> Canvas | Figure | np.ndarray:
    """
    Plot a 3D mesh made of triangles or other polygon.

    Args:
        vertices: The array vertices.
        faces: The array of face indices.

    Returns:
        The resulting plot output.
    """

@draw_mesh.register("vispy")
def _(vertices, faces, *args, **kwargs):
    from vispy import scene
    from vispy.scene.visuals import Mesh

    canvas = scene.SceneCanvas(keys="interactive", bgcolor="white")
    view = canvas.central_widget.add_view()
    view.add(Mesh(vertices, faces, shading="flat", *args, **kwargs))
    
    view.camera = "arcball"
    view.camera.depth_value = 1e3
    
    return canvas

@draw_mesh.register("plotly")
def _(vertices, faces, *args, **kwargs):
    import plotly.graph_objects as go

    x, y, z = vertices.T
    i, j, k = faces.T
    return go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, **kwargs)])
