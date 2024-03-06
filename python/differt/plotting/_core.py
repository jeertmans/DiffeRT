"""Core plotting implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from jaxtyping import Float, Num, UInt

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

    Return:
        The resulting plot output.

    Examples:
        The following example shows how to plot a pyramid mesh.

        .. plotly::

            >>> from differt.plotting import draw_mesh
            >>>
            >>> vertices = np.array(
            ...     [
            ...         [0.0, 0.0, 0.0],
            ...         [1.0, 0.0, 0.0],
            ...         [1.0, 1.0, 0.0],
            ...         [0.0, 1.0, 0.0],
            ...         [0.5, 0.5, 1.0],
            ...     ]
            ... )
            >>> triangles = np.array(
            ...     [[0, 1, 2], [0, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
            ... )
            >>> fig = draw_mesh(vertices, triangles, backend="plotly", opacity=0.5)
            >>> fig  # doctest: +SKIP

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

    Return:
        The resulting plot output.

    Examples:
        The following example shows how to plot ten line strings.

        .. plotly::

            >>> from differt.plotting import draw_paths
            >>>
            >>> def rotation(angle: float) -> np.ndarray:
            ...     c = np.cos(angle)
            ...     s = np.sin(angle)
            ...     return np.array(
            ...         [
            ...             [+c, -s, 0.0],
            ...             [+s, +c, 0.0],
            ...             [0.0, 0.0, 1.0],
            ...         ]
            ...     )
            >>>
            >>> path = np.array(
            ...     [
            ...         [0.0, 0.0, 0.0],
            ...         [1.0, 0.0, 0.0],
            ...         [1.0, 1.0, 0.0],
            ...         [0.1, 0.1, 0.0],
            ...     ],
            ... )
            >>> paths = np.stack(
            ...     [
            ...         path @ rotation(angle) + np.array([0.0, 0.0, 0.1 * dz])
            ...         for dz, angle in enumerate(np.linspace(0, 2 * np.pi, 10))
            ...     ]
            ... )
            >>> fig = draw_paths(
            ...     paths,
            ...     backend="plotly",
            ...     marker=dict(size=0, color="red"),
            ...     line=dict(color="black", width=3),
            ... )
            >>> fig  # doctest: +SKIP
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
            passed to :py:class:`Text<vispy.scene.visuals.Text>`
            if VisPy backend is used.

            By default, ``font_sise=1000`` is used.
        kwargs: Keyword arguments passed to
            :py:class:`Markers<vispy.scene.visuals.Markers>`,
            or :py:class:`Scatter3d<plotly.graph_objects.Scatter3d>`, depending on the
            backend.

    Return:
        The resulting plot output.

    Warning:
        Unsupported backend(s): Matplotlib.

    Examples:
        The following example shows how to plot several annotated markes.

        .. plotly::

            >>> from differt.plotting import draw_markers
            >>>
            >>> markers = np.array(
            ...     [
            ...         [0.0, 0.0, 0.0],
            ...         [1.0, 0.0, 0.0],
            ...         [1.0, 1.0, 0.0],
            ...         [0.0, 1.0, 0.0],
            ...     ]
            ... )
            >>> labels = ["A", "B", "C", "D"]
            >>> fig = draw_markers(markers, labels, backend="plotly")
            >>> fig  # doctest: +SKIP
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


@dispatch  # type: ignore
def draw_image(
    data: Num[np.ndarray, "m n"] | Num[np.ndarray, "m n 3"] | Num[np.ndarray, "m n 4"],
    x: Float[np.ndarray, " *m"] | None = None,
    y: Float[np.ndarray, " *n"] | None = None,
    z0: float = 0.0,
    **kwargs: Any,
) -> Canvas | MplFigure | Figure:  # type: ignore
    """
    Plot a 2D image on a 3D canvas, at using a fixed z-coordinate.

    Args:
        data: The image data array. Can be grayscale, RGB or RGBA.
            For more details on how the data is interpreted, please
            refer to the documentation of the function corresponding
            to the specified backend (see below).
        x: The x-coordinates corresponding to first dimension
            of the image. Those coordinates will be used to scale and translate
            the image.
        y: The y-coordinates corresponding to second dimension
            of the image. Those coordinates will be used to scale and translate
            the image.
        z0: The z-coordinate at which the image is placed.
        kwargs: Keyword arguments passed to
            :py:class:`Mesh<vispy.scene.visuals.Image>`,
            :py:meth:`plot_trisurf<mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface>`,
            or :py:class:`Mesh3d<plotly.graph_objects.Surface>`, depending on the
            backend.

    Return:
        The resulting plot output.

    Warning:
        Matplotlib backend requires ``data`` to be either RGB or RGBA array.

    Examples:
        The following example shows how plot a 2-D image,
        without and with axis scaling.

        .. plotly::
            :fig-vars: fig1, fig2

            >>> from differt.plotting import draw_image
            >>>
            >>> x = np.linspace(-1.0, +1.0, 100)
            >>> y = np.linspace(-4.0, +4.0, 200)
            >>> X, Y = np.meshgrid(x, y)
            >>> Z = np.sin(X) * np.cos(Y)
            >>> fig1 = draw_image(Z, backend="plotly")
            >>> fig1  # doctest: +SKIP
            >>>
            >>> fig2 = draw_image(Z, x=x, y=y, backend="plotly")
            >>> fig2  # doctest: +SKIP

    """


@draw_image.register("vispy")
def _(
    data: Num[np.ndarray, "m n"] | Num[np.ndarray, "m n 3"] | Num[np.ndarray, "m n 4"],
    x: Float[np.ndarray, " ..."] | None = None,
    y: Float[np.ndarray, " ..."] | None = None,
    z0: float = 0.0,
    **kwargs: Any,
) -> Canvas:
    from vispy.scene.visuals import Image
    from vispy.visuals.transforms import STTransform

    canvas, view = process_vispy_kwargs(kwargs)

    if np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)

    image = Image(data, **kwargs)

    m, n = data.shape[:2]

    if x is not None:
        xmin = np.min(x)
        xmax = np.max(x)
        xshift = xmin
        xscale = abs(xmax - xmin) / m
    else:
        xshift = 0.0
        xscale = 1.0

    if y is not None:
        ymin = np.min(y)
        ymax = np.max(y)
        yshift = ymin
        yscale = abs(ymax - ymin) / n
    else:
        yshift = 0.0
        yscale = 1.0

    image.transform = STTransform(
        scale=(xscale, yscale),
        translate=(xshift, yshift, z0),
    )

    view.add(image)

    return canvas


@draw_image.register("matplotlib")
def _(
    data: Num[np.ndarray, "m n"] | Num[np.ndarray, "m n 3"] | Num[np.ndarray, "m n 4"],
    x: Float[np.ndarray, " ..."] | None = None,
    y: Float[np.ndarray, " ..."] | None = None,
    z0: float = 0.0,
    **kwargs: Any,
) -> MplFigure:
    fig, ax = process_matplotlib_kwargs(kwargs)

    ax.plot_surface(X=x, Y=y, Z=np.full_like(data, z0), color=data, **kwargs)

    return fig


@draw_image.register("plotly")
def _(
    data: Num[np.ndarray, "m n"] | Num[np.ndarray, "m n 3"] | Num[np.ndarray, "m n 4"],
    x: Float[np.ndarray, " ..."] | None = None,
    y: Float[np.ndarray, " ..."] | None = None,
    z0: float = 0.0,
    **kwargs: Any,
) -> Figure:
    fig = process_plotly_kwargs(kwargs)

    return fig.add_surface(
        x=x, y=y, z=np.full_like(data, z0), surfacecolor=data, **kwargs
    )
