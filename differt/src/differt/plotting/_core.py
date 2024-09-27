"""Core plotting implementations."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from jaxtyping import Float, Int, Num

from ._utils import (
    dispatch,
    process_matplotlib_kwargs,
    process_plotly_kwargs,
    process_vispy_kwargs,
)

# We cannot use from __future__ import annotations because
#   otherwise array annotations do not render correctly.
# We cannot rely on TYPE_CHECKING-guarded annotation
#   because Sphinx will fail to import this NumPy or Jax typing
# Hence, we prefer to silence pyright instead.

try:
    from matplotlib.figure import Figure as MplFigure
except ImportError:
    MplFigure = Any

try:
    from plotly.graph_objects import Figure
except ImportError:
    Figure = Any

try:
    from vispy.scene.canvas import SceneCanvas as Canvas
except ImportError:
    Canvas = Any


@dispatch
def draw_mesh(
    vertices: Float[np.ndarray, "num_vertices 3"],
    triangles: Int[np.ndarray, "num_triangles 3"],
    **kwargs: Any,
) -> Canvas | MplFigure | Figure:  # type: ignore[reportInvalidTypeForm]
    """
    Plot a 3D mesh made of triangles.

    Args:
        vertices: The array of triangle vertices.
        triangles: The array of triangle indices.
        kwargs: Keyword arguments passed to
            :class:`Mesh<vispy.scene.visuals.Mesh>`,
            :meth:`plot_trisurf<mpl_toolkits.mplot3d.axes3d.Axes3D.plot_trisurf>`,
            or :class:`Mesh3d<plotly.graph_objects.Mesh3d>`, depending on the
            backend.

            .. important::

                If you pass some ``face_colors`` keyword argument,
                it will be passed to Plotly as ``facecolor`` unless
                they were manually passed, in which case
                the ``face_colors`` argument is ignored.
                Matplotlib does not currently support individual
                face colors, so this argument is ignored.

    Returns:
        The resulting plot output.

    Examples:
        The following example shows how to plot a pyramid mesh.

        .. plotly::

            >>> from differt.plotting import draw_mesh
            >>>
            >>> vertices = np.array([
            ...     [0.0, 0.0, 0.0],
            ...     [1.0, 0.0, 0.0],
            ...     [1.0, 1.0, 0.0],
            ...     [0.0, 1.0, 0.0],
            ...     [0.5, 0.5, 1.0],
            ... ])
            >>> triangles = np.array([
            ...     [0, 1, 2],
            ...     [0, 2, 3],
            ...     [0, 1, 4],
            ...     [1, 2, 4],
            ...     [2, 3, 4],
            ...     [3, 0, 4],
            ... ])
            >>> fig = draw_mesh(vertices, triangles, backend="plotly", opacity=0.5)
            >>> fig  # doctest: +SKIP

    """


@draw_mesh.register("vispy")
def _(
    vertices: Float[np.ndarray, "num_vertices 3"],
    triangles: Int[np.ndarray, "num_triangles 3"],
    **kwargs: Any,
) -> Canvas:  # type: ignore[reportInvalidTypeForm]
    from vispy.scene.visuals import Mesh  # noqa: PLC0415

    canvas, view = process_vispy_kwargs(kwargs)

    view.add(Mesh(vertices=vertices, faces=triangles, shading="flat", **kwargs))
    view.camera.set_range()

    return canvas


@draw_mesh.register("matplotlib")
def _(
    vertices: Float[np.ndarray, "num_vertices 3"],
    triangles: Int[np.ndarray, "num_triangles 3"],
    **kwargs: Any,
) -> MplFigure:  # type: ignore[reportInvalidTypeForm]
    fig, ax = process_matplotlib_kwargs(kwargs)

    kwargs.pop("face_colors", None)

    x, y, z = vertices.T
    ax.plot_trisurf(x, y, z, triangles=triangles, **kwargs)

    return fig


@draw_mesh.register("plotly")
def _(
    vertices: Float[np.ndarray, "num_vertices 3"],
    triangles: Int[np.ndarray, "num_triangles 3"],
    **kwargs: Any,
) -> Figure:  # type: ignore[reportInvalidTypeForm]
    fig = process_plotly_kwargs(kwargs)

    if (
        face_colors := kwargs.pop("face_colors", None)
    ) is not None and "facecolor" not in kwargs:
        kwargs["facecolor"] = face_colors

    x, y, z = vertices.T
    i, j, k = triangles.T

    return fig.add_mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, **kwargs)


@dispatch
def draw_paths(
    paths: Float[np.ndarray, "batch path_length 3"],
    **kwargs: Any,
) -> Canvas | MplFigure | Figure:  # type: ignore[reportInvalidTypeForm]
    """
    Plot a batch of paths of the same length.

    Args:
        paths: The array of path vertices.
        kwargs: Keyword arguments passed to
            :class:`LinePlot<vispy.scene.visuals.LinePlot>`,
            :meth:`plot<mpl_toolkits.mplot3d.axes3d.Axes3D.plot>`,
            or :class:`Scatter3d<plotly.graph_objects.Scatter3d>`, depending on the
            backend.

    Returns:
        The resulting plot output.

    Examples:
        The following example shows how to plot ten line strings.

        .. plotly::

            >>> from differt.plotting import draw_paths
            >>>
            >>> def rotation(angle: float) -> np.ndarray:
            ...     co = np.cos(angle)
            ...     si = np.sin(angle)
            ...     return np.array([
            ...         [+co, -si, 0.0],
            ...         [+si, +co, 0.0],
            ...         [0.0, 0.0, 1.0],
            ...     ])
            >>>
            >>> path = np.array(
            ...     [
            ...         [0.0, 0.0, 0.0],
            ...         [1.0, 0.0, 0.0],
            ...         [1.0, 1.0, 0.0],
            ...         [0.1, 0.1, 0.0],
            ...     ],
            ... )
            >>> paths = np.stack([
            ...     path @ rotation(angle) + np.array([0.0, 0.0, 0.1 * dz])
            ...     for dz, angle in enumerate(np.linspace(0, 2 * np.pi, 10))
            ... ])
            >>> fig = draw_paths(
            ...     paths,
            ...     backend="plotly",
            ...     marker=dict(size=0, color="red"),
            ...     line=dict(color="black", width=3),
            ... )
            >>> fig  # doctest: +SKIP
    """


@draw_paths.register("vispy")
def _(
    paths: Float[np.ndarray, "*batch path_length 3"],
    **kwargs: Any,
) -> Canvas:  # type: ignore[reportInvalidTypeForm]
    from vispy.scene.visuals import LinePlot  # noqa: PLC0415

    canvas, view = process_vispy_kwargs(kwargs)

    kwargs.setdefault("width", 3.0)
    kwargs.setdefault("marker_size", 0.0)

    for path in paths.reshape(-1, *paths.shape[-2:]):
        x, y, z = path.T
        view.add(LinePlot(data=(x, y, z), **kwargs))

    view.camera.set_range()

    return canvas


@draw_paths.register("matplotlib")
def _(
    paths: Float[np.ndarray, "*batch path_length 3"],
    **kwargs: Any,
) -> MplFigure:  # type: ignore[reportInvalidTypeForm]
    fig, ax = process_matplotlib_kwargs(kwargs)

    for path in paths.reshape(-1, *paths.shape[-2:]):
        ax.plot(*path.T, **kwargs)

    return fig


@draw_paths.register("plotly")
def _(
    paths: Float[np.ndarray, "*batch path_length 3"],
    **kwargs: Any,
) -> Figure:  # type: ignore[reportInvalidTypeForm]
    fig = process_plotly_kwargs(kwargs)

    for path in paths.reshape(-1, *paths.shape[-2:]):
        x, y, z = path.T
        fig = fig.add_scatter3d(x=x, y=y, z=z, **kwargs)

    return fig


@dispatch
def draw_rays(
    ray_origins: Float[np.ndarray, "*batch 3"],
    ray_directions: Float[np.ndarray, "*batch 3"],
    **kwargs: Any,
) -> Canvas | MplFigure | Figure:  # type: ignore[reportInvalidTypeForm]
    """
    Plot a batch of rays.

    Args:
        ray_origins: An array of origin vertices.
        ray_directions: An array of ray directions. The ray ends
            should be equal to ``ray_origins + ray_directions``.
        kwargs: Keyword arguments passed to
            :class:`LinePlot<vispy.scene.visuals.Arrow>`,
            :meth:`plot<mpl_toolkits.mplot3d.axes3d.Axes3D.quiver>`,
            or :func:`draw_paths` (because VisPy and Plotly don't have a nice quiver plot),
            depending on the backend.

    Returns:
        The resulting plot output.

    Examples:
        The following example shows how to plot rays.

        .. plotly::

            >>> from differt.geometry.utils import fibonacci_lattice
            >>> from differt.plotting import draw_rays
            >>>
            >>> ray_origins = np.zeros(3)
            >>> ray_directions = np.asarray(fibonacci_lattice(50))  # From JAX to NumPy array
            >>> ray_origins, ray_directions = np.broadcast_arrays(ray_origins, ray_directions)
            >>> fig = draw_rays(
            ...     ray_origins,
            ...     ray_directions,
            ...     backend="plotly",
            ... )
            >>> fig  # doctest: +SKIP
    """


@draw_rays.register("vispy")
def _(
    ray_origins: Float[np.ndarray, "*batch 3"],
    ray_directions: Float[np.ndarray, "*batch 3"],
    **kwargs: Any,
) -> Canvas:  # type: ignore[reportInvalidTypeForm]
    ray_ends = ray_origins + ray_directions
    paths = np.concatenate((ray_origins[..., None, :], ray_ends[..., None, :]), axis=-2)

    return draw_paths(paths, backend="vispy", **kwargs)


@draw_rays.register("matplotlib")
def _(
    ray_origins: Float[np.ndarray, "*batch 3"],
    ray_directions: Float[np.ndarray, "*batch 3"],
    **kwargs: Any,
) -> MplFigure:  # type: ignore[reportInvalidTypeForm]
    fig, ax = process_matplotlib_kwargs(kwargs)

    ray_origins = ray_origins.reshape(-1, 3)
    ray_directions = ray_directions.reshape(-1, 3)

    ax.quiver(*ray_origins.T, *ray_directions.T, **kwargs)

    return fig


@draw_rays.register("plotly")
def _(
    ray_origins: Float[np.ndarray, "*batch 3"],
    ray_directions: Float[np.ndarray, "*batch 3"],
    **kwargs: Any,
) -> Figure:  # type: ignore[reportInvalidTypeForm]
    ray_ends = ray_origins + ray_directions
    paths = np.concatenate((ray_origins[..., None, :], ray_ends[..., None, :]), axis=-2)

    return draw_paths(paths, backend="plotly", **kwargs)


@dispatch
def draw_markers(
    markers: Float[np.ndarray, "num_markers 3"],
    labels: Sequence[str] | None = None,
    text_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Canvas | MplFigure | Figure:  # type: ignore[reportInvalidTypeForm]
    """
    Plot markers and, optionally, their label.

    Args:
        markers: The array of marker vertices.
        labels: The marker labels.
        text_kwargs: A mapping of keyword arguments
            passed to :class:`Text<vispy.scene.visuals.Text>`
            if VisPy backend is used.

            By default, ``font_size=1000`` is used.
        kwargs: Keyword arguments passed to
            :class:`Markers<vispy.scene.visuals.Markers>`,
            or :class:`Scatter3d<plotly.graph_objects.Scatter3d>`, depending on the
            backend.

    Returns:
        The resulting plot output.

    Warning:
        Unsupported backend(s): Matplotlib.

    Examples:
        The following example shows how to plot several annotated markers.

        .. plotly::

            >>> from differt.plotting import draw_markers
            >>>
            >>> markers = np.array([
            ...     [0.0, 0.0, 0.0],
            ...     [1.0, 0.0, 0.0],
            ...     [1.0, 1.0, 0.0],
            ...     [0.0, 1.0, 0.0],
            ... ])
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
) -> Canvas:  # type: ignore[reportInvalidTypeForm]
    from vispy.scene.visuals import Markers, Text  # noqa: PLC0415

    canvas, view = process_vispy_kwargs(kwargs)
    kwargs.setdefault("size", 1)
    kwargs.setdefault("edge_width_rel", 0.05)
    kwargs.setdefault("scaling", "scene")
    view.add(Markers(pos=markers, **kwargs))

    if labels:
        text_kwargs = {"font_size": 400, **(text_kwargs or {})}
        view.add(Text(text=labels, pos=markers, **text_kwargs))

    view.camera.set_range()

    return canvas


@draw_markers.register("matplotlib")
def _(
    markers: Float[np.ndarray, "num_markers 3"],
    labels: Sequence[str] | None = None,
    text_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> MplFigure:  # type: ignore[reportInvalidTypeForm]
    raise NotImplementedError  # TODO: implement this


@draw_markers.register("plotly")
def _(
    markers: Float[np.ndarray, "num_markers 3"],
    labels: Sequence[str] | None = None,
    text_kwargs: Mapping[str, Any] | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> Figure:  # type: ignore[reportInvalidTypeForm]
    fig = process_plotly_kwargs(kwargs)

    if labels:
        kwargs = {"mode": "markers+text", **kwargs}
    else:
        kwargs = {"mode": "markers", **kwargs}

    x, y, z = markers.T
    return fig.add_scatter3d(
        x=x,
        y=y,
        z=z,
        text=labels,
        **kwargs,
    )


@dispatch
def draw_image(
    data: Num[np.ndarray, "rows cols"]
    | Num[np.ndarray, "rows cols 3"]
    | Num[np.ndarray, "rows cols 4"],
    x: Float[np.ndarray, " cols"] | None = None,
    y: Float[np.ndarray, " rows"] | None = None,
    z0: float = 0.0,
    **kwargs: Any,
) -> Canvas | MplFigure | Figure:  # type: ignore[reportInvalidTypeForm]
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
            :class:`Mesh<vispy.scene.visuals.Image>`,
            :meth:`plot_trisurf<mpl_toolkits.mplot3d.axes3d.Axes3D.contourf>`,
            or :class:`Mesh3d<plotly.graph_objects.Surface>`, depending on the
            backend.

    Returns:
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
    data: Num[np.ndarray, "rows cols"]
    | Num[np.ndarray, "rows cols 3"]
    | Num[np.ndarray, "rows cols 4"],
    x: Float[np.ndarray, " cols"] | None = None,
    y: Float[np.ndarray, " rows"] | None = None,
    z0: float = 0.0,
    **kwargs: Any,
) -> Canvas:  # type: ignore[reportInvalidTypeForm]
    from vispy.scene.visuals import Image  # noqa: PLC0415
    from vispy.visuals.transforms import STTransform  # noqa: PLC0415

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
    data: Num[np.ndarray, "rows cols"]
    | Num[np.ndarray, "rows cols 3"]
    | Num[np.ndarray, "rows cols 4"],
    x: Float[np.ndarray, " cols"] | None = None,
    y: Float[np.ndarray, " rows"] | None = None,
    z0: float = 0.0,
    **kwargs: Any,
) -> MplFigure:  # type: ignore[reportInvalidTypeForm]
    fig, ax = process_matplotlib_kwargs(kwargs)

    m, n = data.shape[:2]

    if x is None:
        x = np.arange(n)

    if y is None:
        y = np.arange(m)

    ax.contourf(x, y, data, offset=z0, **kwargs)

    return fig


@draw_image.register("plotly")
def _(
    data: Num[np.ndarray, "rows cols"]
    | Num[np.ndarray, "rows cols 3"]
    | Num[np.ndarray, "rows cols 4"],
    x: Float[np.ndarray, " cols"] | None = None,
    y: Float[np.ndarray, " rows"] | None = None,
    z0: float = 0.0,
    **kwargs: Any,
) -> Figure:  # type: ignore[reportInvalidTypeForm]
    fig = process_plotly_kwargs(kwargs)

    return fig.add_surface(
        x=x,
        y=y,
        z=np.full_like(data, z0),
        surfacecolor=data,
        **kwargs,
    )
