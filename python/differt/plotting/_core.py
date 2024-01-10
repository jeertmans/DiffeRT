"""Core plotting implementations."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from jaxtyping import Float, UInt

from ._utils import (
    dispatch,
    process_matplotlib_kwargs,
    process_plotly_kwargs,
    process_vispy_kwargs,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure as MplFigure
    from plotly.graph_objects import Figure
    from vispy.scene.canvas import SceneCanvas as Canvas


@dispatch  # type: ignore
def draw_mesh(
    vertices: Float[np.ndarray, "num_vertices 3"],
    triangles: UInt[np.ndarray, "num_triangles 3"],
    **kwargs: Any,
) -> Canvas | MplFigure | Figure:  # type: ignore
    """
    Plot a 3D mesh made of triangles.

    Args:
        vertices: The array of triangle vertices.
        triangles: The array of triangle indices.
        kwargs: Keyword arguments passed to
            :py:class:`Mesh<vispy.scene.visuals.Mesh>`,
            :py:meth:`plot_trisurf<mpl_toolkits.mplot3d.axes3d.Axes3D.plot_trisurf>`,
            or :py:class:`Mesh3d<plotly.graph_objects.Mesh3d>`, depending on the
            backend.

    Returns:
        The resulting plot output.
    """


@draw_mesh.register("vispy")
def _(
    vertices: Float[np.ndarray, "num_vertices 3"],
    triangles: UInt[np.ndarray, "num_triangles 3"],
    **kwargs: Any,
) -> Canvas:
    from vispy.scene.visuals import Mesh

    canvas, view = process_vispy_kwargs(kwargs)

    view.add(Mesh(vertices=vertices, faces=triangles, shading="flat", **kwargs))
    view.camera.set_range()

    return canvas


@draw_mesh.register("matplotlib")
def _(
    vertices: Float[np.ndarray, "num_vertices 3"],
    triangles: UInt[np.ndarray, "num_triangles 3"],
    **kwargs: Any,
) -> MplFigure:
    fig, ax = process_matplotlib_kwargs(kwargs)

    x, y, z = vertices.T
    ax.plot_trisurf(x, y, z, triangles=triangles, **kwargs)

    return fig


@draw_mesh.register("plotly")
def _(
    vertices: Float[np.ndarray, "num_vertices 3"],
    triangles: UInt[np.ndarray, "num_triangles 3"],
    **kwargs: Any,
) -> Figure:
    fig = process_plotly_kwargs(kwargs)

    x, y, z = vertices.T
    i, j, k = triangles.T

    return fig.add_mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, **kwargs)


@dispatch  # type: ignore
def draw_paths(
    paths: Float[np.ndarray, "*batch path_length 3"], **kwargs: Any
) -> Canvas | MplFigure | Figure:  # type: ignore
    """
    Plot a batch of paths of the same length.

    Args:
        paths: The array of path vertices.
        kwargs: Keyword arguments passed to
            :py:class:`LinePlot<vispy.scene.visuals.LinePlot>`,
            :py:meth:`plot<mpl_toolkits.mplot3d.axes3d.Axes3D.plot>`,
            or :py:class:`Scatter3d<plotly.graph_objects.Scatter3d>`, depending on the
            backend.

    Returns:
        The resulting plot output.
    """


@draw_paths.register("vispy")
def _(paths: Float[np.ndarray, "*batch path_length 3"], **kwargs: Any) -> Canvas:
    from vispy.scene.visuals import LinePlot

    canvas, view = process_vispy_kwargs(kwargs)

    for i in np.ndindex(paths.shape[:-2]):
        view.add(LinePlot(data=paths[i], **kwargs))

    view.camera.set_range()

    return canvas


@draw_paths.register("matplotlib")
def _(paths: Float[np.ndarray, "*batch path_length 3"], **kwargs: Any) -> MplFigure:
    fig, ax = process_matplotlib_kwargs(kwargs)

    for i in np.ndindex(paths.shape[:-2]):
        ax.plot(*paths[i].T, **kwargs)

    return fig


@draw_paths.register("plotly")
def _(paths: Float[np.ndarray, "*batch path_length 3"], **kwargs: Any) -> Figure:
    fig = process_plotly_kwargs(kwargs)

    for i in np.ndindex(paths.shape[:-2]):
        x, y, z = paths[i].T
        fig = fig.add_scatter3d(x=x, y=y, z=z, **kwargs)

    return fig


@dispatch  # type: ignore
def draw_markers(
    markers: Float[np.ndarray, "num_markers 3"],
    labels: Sequence[str] | None = None,
    text_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Canvas | MplFigure | Figure:  # type: ignore
    """
    Plot markers and, optionally, their label.

    Args:
        markers: The array of marker vertices.
        labels: The marker labels.
        text_kwargs: A mapping of keyword arguments
            to be passed to :py:class:`Text<vispy.scene.visuals.Text>`
            if VisPy backend is used.

            By default, ``font_sise=1000`` is used.
        kwargs: Keyword arguments passed to
            :py:class:`Markers<vispy.scene.visuals.Markers>`,
            or :py:class:`Scatter3d<plotly.graph_objects.Scatter3d>`, depending on the
            backend.

    Returns:
        The resulting plot output.

    Warning:
        Unsupported backend(s): Matplotlib.
    """


@draw_markers.register("vispy")
def _(
    markers: Float[np.ndarray, "num_markers 3"],
    labels: Sequence[str] | None = None,
    text_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Canvas:
    from vispy.scene.visuals import Markers, Text

    canvas, view = process_vispy_kwargs(kwargs)
    view.add(Markers(pos=markers, **kwargs))

    if labels:
        text_kwargs = {"font_size": 1000, **(text_kwargs or {})}
        view.add(Text(text=labels, pos=markers, **text_kwargs))

    view.camera.set_range()

    return canvas


@draw_markers.register("matplotlib")
def _(
    markers: Float[np.ndarray, "num_markers 3"],
    labels: Sequence[str] | None = None,
    text_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> MplFigure:
    raise NotImplementedError  # TODO


@draw_markers.register("plotly")
def _(
    markers: Float[np.ndarray, "num_markers 3"],
    labels: Sequence[str] | None = None,
    text_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Figure:
    fig = process_plotly_kwargs(kwargs)

    if labels:
        kwargs = {"mode": "markers+text", **kwargs}

    x, y, z = markers.T
    return fig.add_scatter3d(
        x=x,
        y=y,
        z=z,
        text=labels,
        **kwargs,
    )
