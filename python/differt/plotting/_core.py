"""
Core plotting implementations.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Float, UInt, jaxtyped
from typeguard import typechecked as typechecker

from ._utils import (
    dispatch,
    process_matplotlib_kwargs,
    process_plotly_kwargs,
    process_vispy_kwargs,
)

if TYPE_CHECKING:
    from ._utils import ReturnType


@jaxtyped(typechecker=typechecker)
@dispatch
def draw_mesh(
    vertices: Float[np.ndarray, "num_vertices 3"],
    faces: UInt[np.ndarray, "num_faces num_vertices_per_face"],
    **kwargs,
) -> ReturnType:
    """
    Plot a 3D mesh made of triangles or other polygons.

    Args:
        vertices: The array of vertices.
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

    ax.plot_trisurf(x, y, z, triangles=faces, **kwargs)

    return fig


@draw_mesh.register("plotly")
def _(vertices, faces, *args, **kwargs):
    fig = process_plotly_kwargs(kwargs)

    x, y, z = vertices.T
    i, j, k = faces.T

    return fig.add_mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, **kwargs)


@jaxtyped(typechecker=typechecker)
@dispatch
def draw_paths(
    paths: Float[np.ndarray, "*batch path_length 3"], **kwargs
) -> ReturnType:
    """
    Plot a batch of paths of the same length.

    Args:
        paths: The array of path vertices.

    Returns:
        The resulting plot output.
    """


@draw_paths.register("vispy")
def _(paths, **kwargs):
    from vispy.scene.visuals import LinePlot

    canvas, view = process_vispy_kwargs(kwargs)

    for i in np.ndindex(paths.shape[:-2]):
        view.add(LinePlot(data=paths[i], **kwargs))

    view.camera.set_range()

    return canvas


@draw_paths.register("matplotlib")
def _(paths, **kwargs):
    fig, ax = process_matplotlib_kwargs(kwargs)

    for i in np.ndindex(paths.shape[:-2]):
        ax.plot(*paths[i].T, **kwargs)

    return fig


@draw_paths.register("plotly")
def _(paths, *args, **kwargs):
    fig = process_plotly_kwargs(kwargs)

    for i in np.ndindex(paths.shape[:-2]):
        x, y, z = paths[i].T
        fig = fig.add_scatter3d(x=x, y=y, z=z, **kwargs)

    return fig


@jaxtyped(typechecker=typechecker)
@dispatch
def draw_markers(
    markers: Float[np.ndarray, "num_markers 3"],
    labels: Sequence[str] | None = None,
    **kwargs,
) -> ReturnType:
    """
    Plot markers and, optionally, their label.

    Args:
        markers: The array of marker vertices.
        labels: The marker labels.

    Returns:
        The resulting plot output.
    """


@draw_markers.register("vispy")
def _(markers, labels=None, **kwargs):
    from vispy.scene.visuals import Markers, Text

    canvas, view = process_vispy_kwargs(kwargs)
    view.add(Markers(pos=markers, **kwargs))

    if labels:
        view.add(Text(text=labels, font_size=1000, pos=markers, **kwargs))

    view.camera.set_range()

    return canvas


@draw_markers.register("matplotlib")
def _(markers, labels=None, **kwargs):
    raise NotImplementedError  # TODO


@draw_markers.register("plotly")
def _(markers, labels=None, mode=None, **kwargs):
    fig = process_plotly_kwargs(kwargs)

    x, y, z = markers.T
    return fig.add_scatter3d(
        x=x,
        y=y,
        z=z,
        mode=mode or ("markers+text" if labels else None),
        text=labels,
        **kwargs,
    )
