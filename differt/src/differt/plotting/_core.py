import os
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from jaxtyping import ArrayLike, Float, Int, Real

from ._utils import (
    dispatch,
    process_matplotlib_kwargs,
    process_plotly_kwargs,
    process_vispy_kwargs,
)

# We cannot use from __future__ import annotations because
# otherwise array annotations do not render correctly.
# However, we still import when building docs (online)
# so return type is correctly documented.
if TYPE_CHECKING or "READTHEDOCS" in os.environ:  # pragma: no cover
    from matplotlib.figure import Figure as MplFigure
    from plotly.graph_objects import Figure
    from vispy.scene.canvas import SceneCanvas as Canvas
else:
    MplFigure = Any
    Figure = Any
    Canvas = Any


PlotOutput = Canvas | MplFigure | Figure
"""The output of any plotting function."""


@dispatch
def draw_mesh(
    vertices: Real[ArrayLike, "num_vertices 3"],
    triangles: Int[ArrayLike, "num_triangles 3"],
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
    vertices: Real[ArrayLike, "num_vertices 3"],
    triangles: Int[ArrayLike, "num_triangles 3"],
    **kwargs: Any,
) -> Canvas:
    from vispy.scene.visuals import Mesh  # noqa: PLC0415

    canvas, view = process_vispy_kwargs(kwargs)

    kwargs.setdefault("shading", "flat")

    vertices = np.asarray(vertices)
    triangles = np.asarray(triangles)
    view.add(Mesh(vertices=vertices, faces=triangles, **kwargs))
    view.camera.set_range()

    return canvas


@draw_mesh.register("matplotlib")
def _(
    vertices: Real[ArrayLike, "num_vertices 3"],
    triangles: Int[ArrayLike, "num_triangles 3"],
    **kwargs: Any,
) -> MplFigure:
    fig, ax = process_matplotlib_kwargs(kwargs)

    kwargs.pop("face_colors", None)

    x, y, z = np.asarray(vertices).T
    triangles = np.asarray(triangles)
    ax.plot_trisurf(x, y, z, triangles=triangles, **kwargs)

    return fig


@draw_mesh.register("plotly")
def _(
    vertices: Real[ArrayLike, "num_vertices 3"],
    triangles: Int[ArrayLike, "num_triangles 3"],
    **kwargs: Any,
) -> Figure:
    fig = process_plotly_kwargs(kwargs)

    if (
        face_colors := kwargs.pop("face_colors", None)
    ) is not None and "facecolor" not in kwargs:
        kwargs["facecolor"] = face_colors

    x, y, z = np.asarray(vertices).T
    i, j, k = np.asarray(triangles).T

    return fig.add_mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, **kwargs)


@dispatch
def draw_paths(
    paths: Real[ArrayLike, "*batch path_length 3"],
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

            >>> from differt.geometry import rotation_matrix_along_z_axis as rot
            >>> from differt.plotting import draw_paths
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
            ...     path @ rot(angle) + np.array([0.0, 0.0, 0.1 * dz])
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
    paths: Real[ArrayLike, "*batch path_length 3"],
    **kwargs: Any,
) -> Canvas:
    from vispy.scene.visuals import LinePlot  # noqa: PLC0415

    canvas, view = process_vispy_kwargs(kwargs)

    kwargs.setdefault("width", 3.0)
    kwargs.setdefault("marker_size", 0.0)
    paths = np.asarray(paths)
    path_length = paths.shape[-2]
    paths = paths.reshape(-1, 3)
    connect = np.ones(paths.shape[0], dtype=bool)
    connect[path_length - 1 :: path_length] = False

    view.add(LinePlot(data=paths, connect=connect, **kwargs))  # type: ignore[reportArgumentType]

    view.camera.set_range()

    return canvas


@draw_paths.register("matplotlib")
def _(
    paths: Real[ArrayLike, "*batch path_length 3"],
    **kwargs: Any,
) -> MplFigure:
    fig, ax = process_matplotlib_kwargs(kwargs)

    paths = np.asarray(paths)
    xs, ys, zs = paths.reshape(-1, *paths.shape[-2:]).T

    ax.plot(xs=xs, ys=ys, zs=zs, **kwargs)

    return fig


@draw_paths.register("plotly")
def _(
    paths: Real[ArrayLike, "*batch path_length 3"],
    **kwargs: Any,
) -> Figure:
    fig = process_plotly_kwargs(kwargs)

    paths = np.asarray(paths)
    paths = paths.reshape(-1, *paths.shape[-2:])
    paths = np.concatenate(
        (paths, np.full((paths.shape[0], 1, 3), np.nan, dtype=paths.dtype)), axis=-2
    )
    x, y, z = paths.reshape(-1, 3).T
    return fig.add_scatter3d(x=x, y=y, z=z, **kwargs)


@dispatch
def draw_rays(
    ray_origins: Real[ArrayLike, "*batch 3"],
    ray_directions: Real[ArrayLike, "*batch 3"],
    *,
    ratio: Float[ArrayLike, " "] = 0.1,
    **kwargs: Any,
) -> Canvas | MplFigure | Figure:  # type: ignore[reportInvalidTypeForm]
    """
    Plot a batch of rays.

    Args:
        ray_origins: An array of origin vertices.
        ray_directions: An array of ray directions. The ray ends
            should be equal to ``ray_origins + ray_directions``.
        ratio: The ratio of the arrow head size with respect to the
            corresponding ray length.
        kwargs: Keyword arguments passed to
            :class:`Arrow<vispy.scene.visuals.Arrow>`,
            :meth:`quiver<mpl_toolkits.mplot3d.axes3d.Axes3D.quiver>`,
            or a mix of :class:`Scatter3d<plotly.graph_objects.Scatter3d>`
            and :class:`Cone<plotly.graph_objects.Cone>` [#f1]_ ,
            depending on the backend.

    Returns:
        The resulting plot output.

    .. [#f1] Plotly's 3D quiver plot is just a cone, which does not look like an
        arrow. To improve it, a line is prepended the cone.

    .. raw:: html

        <details>
        <summary><a>Specific keyword rules for Plotly's backend</a></summary>

    The following keyword arguments have a special meaning:

    - ``color``: the color of the rays;
    - ``name``: the name of the rays;

    The following keyword arguments are passed to the line plot:

    - ``legendgroup=name``;
    - ``line_color=color``;
    - ``mode="lines"``;
    - ``showlegend=False``.

    The following keyword arguments are passed to the cone plot (i.e., arrow head):

    - ``colorscale=[color, color]``;
    - ``hoverinfo="name"``;
    - ``legendgroup=name``;
    - ``name=name``;
    - ``showscale=False``;
    - ``showlegend=True``;
    - ``sizemode="raw"``.

    If you wish to override any of the above arguments, or add any additional,
    you can use the following keyword arguments:

    - ``cone_kwargs``: a mapping of keyword arguments passed to :class:`Cone<plotly.graph_objects.Cone>`;
    - ``line_kwargs``: a mapping of keyword arguments passed to :class:`Scatter3d<plotly.graph_objects.Scatter3d>`;

    Other keyword arguments are passed to both constructors.

    .. raw:: html

        </details>

    Examples:
        The following example shows how to plot rays.

        .. plotly::

            >>> from differt.geometry import fibonacci_lattice
            >>> from differt.plotting import draw_rays
            >>>
            >>> ray_origins = jnp.zeros(3)
            >>> ray_directions = fibonacci_lattice(50)
            >>> ray_origins, ray_directions = jnp.broadcast_arrays(
            ...     ray_origins, ray_directions
            ... )
            >>> fig = draw_rays(
            ...     ray_origins,
            ...     ray_directions,
            ...     backend="plotly",
            ... )
            >>> fig  # doctest: +SKIP
    """


@draw_rays.register("vispy")
def _(
    ray_origins: Real[ArrayLike, "*batch 3"],
    ray_directions: Real[ArrayLike, "*batch 3"],
    *,
    ratio: Float[ArrayLike, " "] = 0.1,
    **kwargs: Any,
) -> Canvas:
    from vispy.scene.visuals import Arrow  # noqa: PLC0415

    canvas, view = process_vispy_kwargs(kwargs)

    kwargs.setdefault("width", 3.0)
    kwargs.setdefault("arrow_size", 6.0)

    ray_origins = np.asarray(ray_origins).reshape(-1, 3)
    ray_directions = np.asarray(ray_directions).reshape(-1, 3)
    ratio = np.asarray(ratio)
    body_ends = ray_origins + (1 - ratio) * ray_directions

    pos = np.concatenate((ray_origins, body_ends), axis=-1).reshape(-1, 3)
    connect = np.ones(pos.shape[0], dtype=bool)
    connect[1::2] = False

    arrows_dir = ratio * ray_directions
    arrows_center = body_ends + 0.5 * arrows_dir
    arrows = np.concatenate((body_ends, arrows_center), axis=-1)

    view.add(Arrow(pos=pos, connect=connect, arrows=arrows, **kwargs))  # type: ignore[reportArgumentType]

    view.camera.set_range()

    return canvas


@draw_rays.register("matplotlib")
def _(
    ray_origins: Real[ArrayLike, "*batch 3"],
    ray_directions: Real[ArrayLike, "*batch 3"],
    *,
    ratio: Float[ArrayLike, " "] = 0.1,
    **kwargs: Any,
) -> MplFigure:
    fig, ax = process_matplotlib_kwargs(kwargs)

    kwargs.setdefault("arrow_length_ratio", float(ratio))  # type: ignore[reportCallIssue]

    ray_origins = np.asarray(ray_origins).reshape(-1, 3)
    ray_directions = np.asarray(ray_directions).reshape(-1, 3)

    ax.quiver(*ray_origins.T, *ray_directions.T, **kwargs)

    return fig


@draw_rays.register("plotly")
def _(
    ray_origins: Real[ArrayLike, "*batch 3"],
    ray_directions: Float[ArrayLike, "*batch 3"],
    *,
    ratio: Float[ArrayLike, " "] = 0.1,
    **kwargs: Any,
) -> Figure:
    ray_origins = np.asarray(ray_origins)
    ray_directions = np.asarray(ray_directions)
    ratio = np.asarray(ratio)

    color = kwargs.pop("color", None)
    name = kwargs.pop("name", None)
    line_kwargs = kwargs.pop("line_kwargs", {})
    cone_kwargs = kwargs.pop("cone_kwargs", {})
    line_kwargs = {**kwargs, **line_kwargs}
    cone_kwargs = {**kwargs, **cone_kwargs}

    line_kwargs.setdefault("legendgroup", name)
    if color:  # pragma: no cover
        line_kwargs.setdefault("line_color", color)
    line_kwargs.setdefault("mode", "lines")
    line_kwargs.setdefault("showlegend", False)

    line_ends = ray_origins + (1 - ratio) * ray_directions
    line_paths = np.concatenate(
        (ray_origins[..., None, :], line_ends[..., None, :]), axis=-2
    )

    fig = draw_paths(line_paths, backend="plotly", **line_kwargs)

    if color is None:  # pragma: no cover
        color = fig.layout.template.layout.colorway[0]
        fig.data[-1].line.color = color
    cone_kwargs.setdefault("colorscale", [color, color])
    cone_kwargs.setdefault("hoverinfo", "name")
    cone_kwargs.setdefault("legendgroup", name)
    cone_kwargs.setdefault("name", name)
    cone_kwargs.setdefault("showscale", False)
    cone_kwargs.setdefault("showlegend", True)
    cone_kwargs.setdefault("sizemode", "raw")

    cone_origins = line_ends.reshape(-1, 3)
    cone_directions = (ratio * ray_directions).reshape(-1, 3)

    return fig.add_cone(
        x=cone_origins[:, 0],
        y=cone_origins[:, 1],
        z=cone_origins[:, 2],
        u=cone_directions[:, 0],
        v=cone_directions[:, 1],
        w=cone_directions[:, 2],
        **cone_kwargs,
    )


@dispatch
def draw_markers(
    markers: Real[ArrayLike, "*batch 3"],
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
            :meth:`scatter<mpl_toolkits.mplot3d.axes3d.Axes3D.scatter>`,
            or :class:`Scatter3d<plotly.graph_objects.Scatter3d>`, depending on the
            backend.

    Returns:
        The resulting plot output.

    Warning:
        Matplotlib does not support labels.

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
    markers: Real[ArrayLike, "*batch 3"],
    labels: Sequence[str] | None = None,
    text_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Canvas:
    from vispy.scene.visuals import Markers, Text  # noqa: PLC0415

    canvas, view = process_vispy_kwargs(kwargs)
    kwargs.setdefault("size", 1)
    kwargs.setdefault("edge_width_rel", 0.05)
    kwargs.setdefault("scaling", "scene")
    markers = np.asarray(markers).reshape(-1, 3)
    view.add(Markers(pos=markers, **kwargs))

    if labels:
        text_kwargs = {"font_size": 400, **(text_kwargs or {})}
        view.add(Text(text=labels, pos=markers, **text_kwargs))

    view.camera.set_range()

    return canvas


@draw_markers.register("matplotlib")
def _(
    markers: Real[ArrayLike, "*batch 3"],
    labels: Sequence[str] | None = None,
    text_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> MplFigure:
    fig, ax = process_matplotlib_kwargs(kwargs)

    if labels is not None:
        msg = "Matplotlib does not currently support adding labels to markers, this option is ignored."
        warnings.warn(msg, UserWarning, stacklevel=2)
        del labels, text_kwargs

    xs, ys, zs = np.asarray(markers).reshape(-1, 3).T

    ax.scatter(xs, ys, zs=zs, **kwargs)

    return fig


@draw_markers.register("plotly")
def _(
    markers: Real[ArrayLike, "*batch 3"],
    labels: Sequence[str] | None = None,
    text_kwargs: Mapping[str, Any] | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> Figure:
    fig = process_plotly_kwargs(kwargs)

    if labels:
        kwargs = {"mode": "markers+text", **kwargs}
    else:
        kwargs = {"mode": "markers", **kwargs}

    x, y, z = np.asarray(markers).reshape(-1, 3).T
    return fig.add_scatter3d(
        x=x,
        y=y,
        z=z,
        text=labels,
        **kwargs,
    )


@dispatch
def draw_image(
    data: Real[ArrayLike, "rows cols"]
    | Real[ArrayLike, "rows cols 3"]
    | Real[ArrayLike, "rows cols 4"],
    x: Real[ArrayLike, " cols"] | Real[ArrayLike, "rows cols 3"] | None = None,
    y: Real[ArrayLike, " rows"] | Real[ArrayLike, "rows cols 3"] | None = None,
    z0: float = 0.0,
    **kwargs: Any,
) -> Canvas | MplFigure | Figure:  # type: ignore[reportInvalidTypeForm]
    """
    Plot a 2D image on a 3D canvas, at a fixed z-coordinate.

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
            :class:`Image<vispy.scene.visuals.Image>`,
            :meth:`contourf<mpl_toolkits.mplot3d.axes3d.Axes3D.contourf>`,
            or :class:`Mesh3d<plotly.graph_objects.Surface>`, depending on the
            backend.

    Returns:
        The resulting plot output.

    Warning:
        Matplotlib backend requires ``data`` to be either RGB or RGBA array.

    Examples:
        The following example shows how to plot a 2-D image,
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
    data: Real[ArrayLike, "rows cols"]
    | Real[ArrayLike, "rows cols 3"]
    | Real[ArrayLike, "rows cols 4"],
    x: Real[ArrayLike, " cols"] | Real[ArrayLike, "rows cols 3"] | None = None,
    y: Real[ArrayLike, " rows"] | Real[ArrayLike, "rows cols 3"] | None = None,
    z0: float = 0.0,
    **kwargs: Any,
) -> Canvas:
    from vispy.scene.visuals import Image  # noqa: PLC0415
    from vispy.visuals.transforms import STTransform  # noqa: PLC0415

    canvas, view = process_vispy_kwargs(kwargs)

    data = np.asarray(data)
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
    data: Real[ArrayLike, "rows cols"]
    | Real[ArrayLike, "rows cols 3"]
    | Real[ArrayLike, "rows cols 4"],
    x: Real[ArrayLike, " cols"] | Real[ArrayLike, "rows cols 3"] | None = None,
    y: Real[ArrayLike, " rows"] | Real[ArrayLike, "rows cols 3"] | None = None,
    z0: float = 0.0,
    **kwargs: Any,
) -> MplFigure:
    fig, ax = process_matplotlib_kwargs(kwargs)

    data = np.asarray(data)
    m, n = data.shape[:2]

    x = np.arange(n) if x is None else np.asarray(x)

    y = np.arange(m) if y is None else np.asarray(y)

    ax.contourf(x, y, data, offset=z0, **kwargs)

    return fig


@draw_image.register("plotly")
def _(
    data: Real[ArrayLike, "rows cols"]
    | Real[ArrayLike, "rows cols 3"]
    | Real[ArrayLike, "rows cols 4"],
    x: Real[ArrayLike, " cols"] | Real[ArrayLike, "rows cols 3"] | None = None,
    y: Real[ArrayLike, " rows"] | Real[ArrayLike, "rows cols 3"] | None = None,
    z0: float = 0.0,
    **kwargs: Any,
) -> Figure:
    fig = process_plotly_kwargs(kwargs)

    data = np.asarray(data)
    x = None if x is None else np.asarray(x)
    y = None if y is None else np.asarray(y)

    return fig.add_surface(
        x=x,
        y=y,
        z=np.full_like(data, z0),
        surfacecolor=data,
        **kwargs,
    )


@dispatch
def draw_contour(  # noqa: PLR0917
    data: Real[ArrayLike, "rows cols"],
    x: Real[ArrayLike, " cols"] | Real[ArrayLike, "rows cols 3"] | None = None,
    y: Real[ArrayLike, " rows"] | Real[ArrayLike, "rows cols 3"] | None = None,
    z0: float = 0.0,
    levels: int | Real[ArrayLike, " num_levels"] | None = None,
    fill: bool = False,
    **kwargs: Any,
) -> Canvas | MplFigure | Figure:  # type: ignore[reportInvalidTypeForm]
    """
    Plot a 2D contour on a 3D canvas, at a fixed z-coordinate.

    Args:
        data: The values over which the contour is drawn.
        x: The x-coordinates corresponding to first dimension
            of the contour. Those coordinates will be used to scale and translate
            the contour.
        y: The y-coordinates corresponding to second dimension
            of the contour. Those coordinates will be used to scale and translate
            the contour.
        z0: The z-coordinate at which the contour is placed.
        levels: The levels at which the contour is drawn.
        fill: Whether to fill the contour.
        kwargs: Keyword arguments passed to
            :class:`Isocurve<vispy.scene.visuals.Isocurve>`,
            :meth:`contour<mpl_toolkits.mplot3d.axes3d.Axes3D.contour>`,
            (or :meth:`contourf<mpl_toolkits.mplot3d.axes3d.Axes3D.contourf>`
            if ``fill`` is :data:`True`),
            or :class:`Contour<plotly.graph_objects.Contour>`, depending on the
            backend.

    Returns:
        The resulting plot output.

    Warning:
        VisPy does not support filling the contour.
        Plotly does not support 3D contours, and will draw the coontour on
        a 2D figure instead.

    Examples:
        The following example shows how to plot a 2-D contour,
        without and with axis scaling, and filling.

        .. plotly::
            :fig-vars: fig1, fig2

            >>> from differt.plotting import draw_contour
            >>>
            >>> x = np.linspace(-1.0, +1.0, 10)
            >>> y = np.linspace(-4.0, +4.0, 20)
            >>> X, Y = np.meshgrid(x, y)
            >>> Z = np.cos(X) * np.sin(Y)
            >>> fig1 = draw_contour(Z, backend="plotly")
            >>> fig1  # doctest: +SKIP
            >>>
            >>> fig2 = draw_contour(Z, x=x, y=y, fill=True, backend="plotly")
            >>> fig2  # doctest: +SKIP

    """


@draw_contour.register("vispy")
def _(  # noqa: PLR0917
    data: Real[ArrayLike, "rows cols"],
    x: Real[ArrayLike, " cols"] | Real[ArrayLike, "rows cols"] | None = None,
    y: Real[ArrayLike, " rows"] | Real[ArrayLike, "rows cols"] | None = None,
    z0: float = 0.0,
    levels: int | Real[ArrayLike, " num_levels"] | None = None,
    fill: bool = False,
    **kwargs: Any,
) -> Canvas:
    from vispy.scene.visuals import Isocurve  # noqa: PLC0415
    from vispy.visuals.transforms import STTransform  # noqa: PLC0415

    data = np.asarray(data)

    if isinstance(levels, int):
        msg = (
            f"VisPy does not support using {type(levels)} as parameters for `levels`. "
            f"A range of {levels + 1 = } values from {data.min() = } to {data.max() = } is used instead."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        levels = np.linspace(data.min(), data.max(), levels + 1)
    else:
        levels = np.asarray(levels)

    if fill:
        msg = "VisPy does not support filling contour, this option is ignored."
        warnings.warn(msg, UserWarning, stacklevel=2)

    canvas, view = process_vispy_kwargs(kwargs)

    iso = Isocurve(data, levels=levels, **kwargs)

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

    # TODO: fix me, this does work

    iso.transform = STTransform(
        scale=(xscale, yscale),
        translate=(xshift, yshift, z0),
    )

    view.add(iso)

    return canvas


@draw_contour.register("matplotlib")
def _(  # noqa: PLR0917
    data: Real[ArrayLike, "rows cols"],
    x: Real[ArrayLike, " cols"] | Real[ArrayLike, "rows cols"] | None = None,
    y: Real[ArrayLike, " rows"] | Real[ArrayLike, "rows cols"] | None = None,
    z0: float = 0.0,
    levels: int | Real[ArrayLike, " num_levels"] | None = None,
    fill: bool = False,
    **kwargs: Any,
) -> MplFigure:
    fig, ax = process_matplotlib_kwargs(kwargs)

    data = np.asarray(data)
    m, n = data.shape[:2]

    x = np.arange(n) if x is None else np.asarray(x)

    y = np.arange(m) if y is None else np.asarray(y)

    if not isinstance(levels, int) and isinstance(levels, ArrayLike):
        levels = np.asarray(levels)

    if fill:
        ax.contourf(x, y, data, offset=z0, levels=levels, **kwargs)
    else:
        ax.contour(x, y, data, offset=z0, levels=levels, **kwargs)

    return fig


@draw_contour.register("plotly")
def _(  # noqa: PLR0917
    data: Real[ArrayLike, "rows cols"],
    x: Real[ArrayLike, " cols"] | Real[ArrayLike, "rows cols"] | None = None,
    y: Real[ArrayLike, " rows"] | Real[ArrayLike, "rows cols"] | None = None,
    z0: float = 0.0,
    levels: int | Real[ArrayLike, " num_levels"] | None = None,
    fill: bool = False,
    **kwargs: Any,
) -> Figure:
    fig = process_plotly_kwargs(kwargs)

    del z0

    if "contours_coloring" not in kwargs and not fill:
        kwargs["contours_coloring"] = "lines"

    if isinstance(levels, int):
        kwargs.setdefault("autocontour", True)
        kwargs.setdefault("ncontours", levels)
    elif isinstance(levels, ArrayLike):
        levels = np.asarray(levels)
        msg = (
            "Plotly does not support arbitrary level values, but only linearly spaced levels. "
            f"A range of values from {levels.min() = } to {levels.max() = } with step "
            f"{(levels.max() - levels.min()) / max(1, levels.size - 1) = :.2f} is used instead."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        contours = kwargs.setdefault("contours", {})
        contours["start"] = levels.min()
        contours["end"] = levels.max()
        contours["size"] = (levels.max() - levels.min()) / max(1, levels.size - 1)

    data = np.asarray(data)
    x = None if x is None else np.asarray(x)
    y = None if y is None else np.asarray(y)

    return fig.add_contour(
        x=x,
        y=y,
        z=data,
        **kwargs,
    )


@dispatch
def draw_surface(
    x: Real[ArrayLike, " cols"] | Real[ArrayLike, "rows cols"] | None = None,
    y: Real[ArrayLike, " rows"] | Real[ArrayLike, "rows cols"] | None = None,
    *,
    z: Real[ArrayLike, "rows cols"],
    colors: Real[ArrayLike, "rows cols"] | Real[ArrayLike, "rows cols 3"] | None = None,
    **kwargs: Any,
) -> Canvas | MplFigure | Figure:  # type: ignore[reportInvalidTypeForm]
    """
    Plot a 3D surface.

    Args:
        x: The x-coordinates corresponding to first dimension
            of the surface.
        y: The y-coordinates corresponding to second dimension
            of the surface.
        z: The z-coordinates corresponding to third dimension
            of the surface.
        colors: The color of values to use.

            In the Plotly backend, the default is to use the values in ``z``.
        kwargs: Keyword arguments passed to
            :class:`Isocurve<vispy.scene.visuals.SurfacePlot>`,
            :meth:`contour<mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface>`,
            or :class:`Surface<plotly.graph_objects.Surface>`, depending on the
            backend.

    Returns:
        The resulting plot output.

    Warning:
        Matplotlib requires ``colors`` to be RGB or RGBA values.
        VisPy currently does not support colors.

    Examples:
        The following example shows how to plot a 3-D surface,
        without and with custom coloring.

        .. plotly::
            :fig-vars: fig1, fig2

            >>> from differt.plotting import draw_surface
            >>>
            >>> u = np.linspace(0, 2 * np.pi, 100)
            >>> v = np.linspace(0, np.pi, 100)
            >>> x = np.outer(np.cos(u), np.sin(v))
            >>> y = np.outer(np.sin(u), np.sin(v))
            >>> z = np.outer(np.cos(u), np.cos(v))
            >>> fig1 = draw_surface(x, y, z=z, backend="plotly")
            >>> fig1  # doctest: +SKIP
            >>>
            >>> fig2 = draw_surface(
            ...     x, y, z=z, colors=x * x + y * y + z * z, backend="plotly"
            ... )
            >>> fig2  # doctest: +SKIP

    """


@draw_surface.register("vispy")
def _(
    x: Real[ArrayLike, " cols"] | Real[ArrayLike, "rows cols"] | None = None,
    y: Real[ArrayLike, " rows"] | Real[ArrayLike, "rows cols"] | None = None,
    *,
    z: Real[ArrayLike, "rows cols"],
    colors: Real[ArrayLike, "rows cols"] | Real[ArrayLike, "rows cols 3"] | None = None,
    **kwargs: Any,
) -> Canvas:
    from vispy.scene.visuals import SurfacePlot  # noqa: PLC0415

    canvas, view = process_vispy_kwargs(kwargs)

    x = None if x is None else np.asarray(x)
    y = None if y is None else np.asarray(y)
    z = np.asarray(z)

    if colors is not None:
        msg = "VisPy does not currently support coloring like we would like."
        warnings.warn(msg, UserWarning, stacklevel=2)
        colors = None

    view.add(SurfacePlot(x=x, y=y, z=z, color=colors, **kwargs))

    return canvas


@draw_surface.register("matplotlib")
def _(
    x: Real[ArrayLike, " cols"] | Real[ArrayLike, "rows cols"] | None = None,
    y: Real[ArrayLike, " rows"] | Real[ArrayLike, "rows cols"] | None = None,
    *,
    z: Real[ArrayLike, "rows cols"],
    colors: Real[ArrayLike, "rows cols"]
    | Real[ArrayLike, "rows cols 3"]
    | Real[ArrayLike, "rows cols 4"]
    | None = None,
    **kwargs: Any,
) -> MplFigure:
    fig, ax = process_matplotlib_kwargs(kwargs)

    z = np.asarray(z)

    x = np.arange(z.shape[1]) if x is None else np.asarray(x)

    if x.ndim == 1:
        x = np.broadcast_to(x[None, :], z.shape)

    y = np.arange(z.shape[0]) if y is None else np.asarray(y)

    if y.ndim == 1:
        y = np.broadcast_to(y[:, None], z.shape)

    if colors is not None and "facecolors" not in kwargs:
        colors = np.asarray(colors)
        if colors.ndim != 3:  # noqa: PLR2004
            msg = "Matplotlib requires 'colors' to be RGB or RGBA values."
            warnings.warn(msg, UserWarning, stacklevel=2)
            c_min = colors.min()
            c_max = colors.max()
            colors = np.broadcast_to(colors[..., None], (*colors.shape, 3))
            colors = (colors - c_min) / (c_max - c_min)
        kwargs["facecolors"] = colors

    ax.plot_surface(x, y, z, **kwargs)

    return fig


@draw_surface.register("plotly")
def _(
    x: Real[ArrayLike, " cols"] | Real[ArrayLike, "rows cols"] | None = None,
    y: Real[ArrayLike, " rows"] | Real[ArrayLike, "rows cols"] | None = None,
    *,
    z: Real[ArrayLike, "rows cols"],
    colors: Real[ArrayLike, "rows cols"] | Real[ArrayLike, "rows cols 3"] | None = None,
    **kwargs: Any,
) -> Figure:
    fig = process_plotly_kwargs(kwargs)

    x = None if x is None else np.asarray(x)
    y = None if y is None else np.asarray(y)
    z = np.asarray(z)

    if colors is not None and "surfacecolor" not in kwargs:
        kwargs["surfacecolor"] = np.asarray(colors)

    fig.add_surface(x=x, y=y, z=z, **kwargs)

    return fig
