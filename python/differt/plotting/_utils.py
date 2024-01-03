"""
Useful decorators for plotting.
"""

from __future__ import annotations

import importlib
from collections.abc import MutableMapping
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Generic, Protocol, TypeVar

CURRENT_BACKEND = None
DEFAULT_BACKEND = "vispy"
SUPPORTED_BACKENDS = ("vispy", "matplotlib", "plotly")

if TYPE_CHECKING:
    from typing import ParamSpec

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure as MplFigure
    from plotly.graph_objects import Figure
    from vispy.scene.canvas import SceneCanvas
    from vispy.scene.widgets.viewbox import ViewBox

    ReturnType = SceneCanvas | MplFigure | Figure

    P = ParamSpec("P")
    T = TypeVar("T", SceneCanvas, MplFigure, Figure)
else:
    P = TypeVar("P")
    T = TypeVar("T")


def use(backend: str) -> None:
    """
    Tell future plotting utilities to use this backend by default.

    Args:
        backend: The name of the backend to use.

    Raises:
        ValueError: If the backend is not supported.
        ImportError: If the backend is not installed.

    Examples:
        The following example shows how to set the default plotting backend.

        >>> import differt.plotting as dplt
        >>>
        >>> @dplt.dispatch
        ... def my_plot():
        ...     pass
        ...
        >>>
        >>> @my_plot.register("vispy")
        ... def _():
        ...     print("Using vispy backend")
        ...
        >>>
        >>> @my_plot.register("matplotlib")
        ... def _():
        ...     print("Using matplotlib backend")
        ...
        >>>
        >>> my_plot()  # When not specified, use default backend
        Using vispy backend
        >>>
        >>> my_plot(backend="matplotlib")  # We can force the backend
        Using matplotlib backend
        >>>
        >>> dplt.use("matplotlib")  # Or change the default backend...
        >>>
        >>> my_plot()  # So that now it defaults to 'matplotlib'
        Using matplotlib backend
        >>>
        >>> my_plot(
        ...     backend="vispy"
        ... )  # Of course, the 'vispy' backend is still available
        Using vispy backend
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"The backend '{backend}' is not supported. "
            f"We currently support: {', '.join(SUPPORTED_BACKENDS)}."
        )

    try:
        importlib.import_module(f"{backend}")
        global DEFAULT_BACKEND
        DEFAULT_BACKEND = backend
    except ImportError:
        raise ImportError(
            f"Could not load backend '{backend}', did you install it?"
        ) from None


class Dispatcher(Protocol, Generic[P, T]):
    def __call__(
        self, *args: P.args, backend: str | None = None, **kwargs: P.kwargs
    ) -> T:
        ...

    def register(
        self,
        backend: str,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        ...

    def dispatch(self, backend: str) -> Callable[P, T]:
        ...


def dispatch(fun: Callable[P, T]) -> Dispatcher[P, T]:
    """
    Transform a function into a backend dispatcher for plot functions.

    Examples:
        The following example shows how one can implement plotting
        utilities on different backends for a given plot.

        >>> import differt.plotting as dplt
        >>>
        >>> @dplt.dispatch
        ... def plot_line(vertices, color):
        ...     pass
        ...
        >>>
        >>> @plot_line.register("matplotlib")
        ... def _(vertices, color):
        ...     print("Using matplotlib backend")
        ...
        >>>
        >>> @plot_line.register("plotly")
        ... def _(vertices, color):
        ...     print("Using plotly backend")
        ...
        >>>
        >>> plot_line(_, _, backend="matplotlib")
        Using matplotlib backend
        >>>
        >>> plot_line(_, _, backend="plotly")
        Using plotly backend
        >>>
        >>> plot_line(_, _, backend="vispy")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        NotImplementedError: No backend implementation for 'vispy'
        >>>
        >>> # The default backend is VisPy so unimplemented too.
        >>> plot_line(_, _)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        NotImplementedError: No backend implementation for 'vispy'
        >>>
        >>> @plot_line.register("numpy")  # doctest: +IGNORE_EXCEPTION_DETAIL
        ... def _(vertices, color):
        ...     pass
        ...
        Traceback (most recent call last):
        ValueError: Unsupported backend 'numpy', allowed values are: ...
    """
    registry = {}

    def register(
        backend: str,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """Register a new implemenation."""
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}', "
                f"allowed values are: {', '.join(SUPPORTED_BACKENDS)}."
            )

        def wrapper(impl: Callable[P, T]) -> Callable[P, T]:
            """Actually register the backend implemention."""

            @wraps(impl)
            def __wrapper__(*args: P.args, **kwargs: P.kwargs) -> T:  # noqa: N807
                try:
                    return impl(*args, **kwargs)
                except ImportError as e:
                    raise ImportError(
                        "An import error occured when dispatching "
                        f"plot utility to backend '{backend}'. "
                        "Did you correctly install it?"
                    ) from e

            registry[backend] = impl

        return wrapper

    def dispatch(backend: str) -> Callable[P, T]:
        try:
            return registry[backend]
        except KeyError:
            raise NotImplementedError(
                f"No backend implementation for '{backend}'"
            ) from None

    @wraps(fun)
    def main_wrapper(
        *args: P.args, backend: str | None = None, **kwargs: P.kwargs
    ) -> T:
        return dispatch(backend or DEFAULT_BACKEND)(*args, **kwargs)

    main_wrapper.register = register  # type: ignore[attr-defined]
    main_wrapper.dispatch = dispatch  # type: ignore[attr-defined]
    main_wrapper.registry = registry  # type: ignore[attr-defined]

    return main_wrapper  # type: ignore[return-value]


def view_from_canvas(canvas: SceneCanvas) -> ViewBox:
    """
    Return the view from the specified canvas.

    If the canvas does not have any view, create one and
    return it.

    This utility is used by :py:func:`process_vispy_kwargs`.

    Args:
        canvas: The canvas that draws the contents of the scene.

    Returns:
        The view on which contents are displayed.
    """
    from vispy.scene.widgets.viewbox import ViewBox

    def default_view() -> ViewBox:
        view = canvas.central_widget.add_view()
        view.camera = "turntable"
        view.camera.depth_value = 1e3
        return view

    return (
        next(
            (
                child
                for child in canvas.central_widget.children
                if isinstance(child, ViewBox)
            ),
            None,
        )
        or default_view()
    )


def process_vispy_kwargs(
    kwargs: MutableMapping[str, Any],
) -> tuple[SceneCanvas, ViewBox]:
    """
    Process keyword arguments passed to some VisPy plotting utility.

    Args:
        kwargs: A mutable mapping of keyword arguments passed to VisPy plotting.

            .. warning::

                The keys specified below will be removed from the mapping.

    Keyword Args:
        convas (:py:class:`SceneCanvas<vispy.scene.canvas.SceneCanvas>`):
            The canvas that draws contents of the scene. If not provided,
            will try to access canvas from ``view`` (if supplied).
        view (:py:class:`Viewbox<vispy.scene.widgets.viewbox.ViewBox>`):
            The view on which contents are displayed. If not provided,
            will try to get a view from ``canvas``
            (if supplied and has at least one view in its children).

    Warning:
        When supplying both ``canvas`` and ``view``, user
        must ensure that ``view in canvas.central_widget.children``
        evaluates to :py:data:`True`.

    Returns:
        The canvas and view used to display contents.
    """
    from vispy import scene

    maybe_view = kwargs.pop("view", None)
    canvas = (
        kwargs.pop("canvas", None)
        or (maybe_view.parent.canvas if maybe_view else None)
        or scene.SceneCanvas(keys="interactive", bgcolor="white")
    )

    view = maybe_view or view_from_canvas(canvas)

    return canvas, view


def process_matplotlib_kwargs(
    kwargs: MutableMapping[str, Any],
) -> tuple[MplFigure, Axes]:
    """
    Process keyword arguments passed to some Matplotlib plotting utility.

    Args:
        kwargs: A mutable mapping of keyword arguments passed to
            Matplotlib plotting.

            .. warning::

                The keys specified below will be removed from the mapping.

    Keyword Args:
        figure (:py:class:`Figure<matplotlib.figure.Figure>`):
            The figure that draws contents of the scene. If not provided,
            will try to access figure from ``ax`` (if supplied).
        ax (:py:class:`Axes<matplotlib.axes.Axes>`):
            The view on which contents are displayed. If not provided,
            will try to get axes from ``figure``
            (if supplied). The default axes will use a 3D projection.

    Warning:
        When supplying both ``figure`` and ``ax``, user
        must ensure that ``ax in figure.axes``
        evaluates to :py:data:`True`.

    Returns:
        The figure and axes used to display contents.
    """
    import matplotlib.pyplot as plt

    maybe_ax = kwargs.pop("ax", None)
    figure = (
        kwargs.pop("figure", None)
        or (maybe_ax.get_figure() if maybe_ax else None)
        or plt.figure()
    )

    ax = (
        maybe_ax
        or (figure.gca() if len(figure.axes) > 0 else None)
        or figure.add_subplot(projection="3d")
    )

    return figure, ax


def process_plotly_kwargs(
    kwargs: MutableMapping[str, Any],
) -> Figure:
    """
    Process keyword arguments passed to some Plotly plotting utility.

    Args:
        kwargs: A mutable mapping of keyword arguments passed to
            Plotly plotting.

            .. warning::

                The keys specified below will be removed from the mapping.

    Keyword Args:
        figure (:py:class:`Figure<plotly.graph_objects.Figure>`):
            The figure that draws contents of the scene.

    Returns:
        The figure used to display contents.
    """
    import plotly.graph_objects as go

    return kwargs.pop("figure", None) or go.Figure()
