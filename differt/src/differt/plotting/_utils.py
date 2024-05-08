"""Useful decorators for plotting."""

import importlib
import sys
import types
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from functools import wraps
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, TypeVar, Union

# Immutables

SUPPORTED_BACKENDS = ("vispy", "matplotlib", "plotly")
"""The list of supported backends."""

BACKEND_LOCK = Lock()
"""A Lock to avoid modifying backend (and defaults) in multiple threads at the same time (e.g., with Pytest."""

# Mutables

DEFAULT_BACKEND = "vispy"
"""The default backend."""
DEFAULT_KWARGS: MutableMapping[str, Any] = {}
"""The default keyword arguments."""


if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

P = ParamSpec("P")

if TYPE_CHECKING:
    from matplotlib.figure import Figure as MplFigure
    from mpl_toolkits.mplot3d import Axes3D
    from plotly.graph_objects import Figure
    from vispy.scene.canvas import SceneCanvas as Canvas
    from vispy.scene.widgets.viewbox import ViewBox
else:
    MplFigure = Any
    Axes3D = Any
    Figure = Any
    Canvas = Any
    ViewBox = Any

T = TypeVar("T", Canvas, MplFigure, Figure)


def set_defaults(backend: Optional[str] = None, **kwargs: Any) -> str:
    """
    Set default keyword arguments for future plotting utilities.

    Args:
        backend: The name of the backend to use, or
            :py:data:`None` to use the current default.
        kwargs: Keyword arguments that will be passed to the
            corresponding ``process_*_kwargs`` function, and
            plot utilities.

    Return:
        The name of the (new) default backend.

    Raises:
        ValueError: If the backend is not supported.
        ImportError: If the backend is not installed.

    Examples:
        The following example shows how to set the default plotting backend
        and other plotting defaults.

        >>> import differt.plotting as dplt
        >>>
        >>> @dplt.dispatch
        ... def my_plot(*args, **kwargs):
        ...     pass
        >>>
        >>> @my_plot.register("vispy")
        ... def _(*args, **kwargs):
        ...     dplt.process_vispy_kwargs(kwargs)
        ...     print(f"Using vispy backend with {args = }, {kwargs = }")
        >>>
        >>> @my_plot.register("matplotlib")
        ... def _(*args, **kwargs):
        ...     dplt.process_matplotlib_kwargs(kwargs)
        ...     print(f"Using matplotlib backend with {args = }, {kwargs = }")
        >>>
        >>> my_plot()  # When not specified, use default backend
        Using vispy backend with args = (), kwargs = {}
        >>>
        >>> dplt.set_defaults("matplotlib")  # We can change the default backend
        'matplotlib'
        >>> my_plot()  # So that now it defaults to 'matplotlib'
        Using matplotlib backend with args = (), kwargs = {}
        >>>
        >>> dplt.set_defaults(
        ...     "matplotlib", color="red"
        ... )  # We can also specify additional defaults
        'matplotlib'
        >>> my_plot()  # So that now it defaults to 'matplotlib' and color='red'
        Using matplotlib backend with args = (), kwargs = {'color': 'red'}
        >>> my_plot(backend="vispy")  # Of course, the 'vispy' backend is still available
        Using vispy backend with args = (), kwargs = {'color': 'red'}
        >>> my_plot(backend="vispy", color="green")  # And we can also override any default
        Using vispy backend with args = (), kwargs = {'color': 'green'}
        >>> dplt.set_defaults("vispy")  # Reset all defaults
        'vispy'
    """
    global DEFAULT_BACKEND, DEFAULT_KWARGS

    if backend is None:
        backend = DEFAULT_BACKEND
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"The backend '{backend}' is not supported. "
            f"We currently support: {', '.join(SUPPORTED_BACKENDS)}."
        )

    try:
        importlib.import_module(f"{backend}")
        DEFAULT_BACKEND = backend
        DEFAULT_KWARGS = kwargs
        return backend
    except ImportError:
        raise ImportError(
            f"Could not load backend '{backend}', did you install it?"
        ) from None


@contextmanager
def use(*args: Any, **kwargs: Any) -> Iterator[str]:
    """
    Create a context manager that sets plotting defaults and returns the current default backend.

    When exiting the context, the previous default backend
    and default keyword arguments are set back.

    Args:
        args: Positional arguments passed to
            :py:func:`set_defaults`.
        kwargs: Keywords arguments passed to
            :py:func:`set_defaults`.

    Return:
        The name of the default backend used in this context.

    Examples:
        The following example shows how set plot defaults in a context.

        >>> import differt.plotting as dplt
        >>>
        >>> @dplt.dispatch
        ... def my_plot(*args, **kwargs):
        ...     pass
        >>>
        >>> @my_plot.register("vispy")
        ... def _(*args, **kwargs):
        ...     dplt.process_vispy_kwargs(kwargs)
        ...     print(f"Using vispy backend with {args = }, {kwargs = }")
        >>>
        >>> @my_plot.register("plotly")
        ... def _(*args, **kwargs):
        ...     dplt.process_plotly_kwargs(kwargs)
        ...     print(f"Using plotly backend with {args = }, {kwargs = }")
        >>>
        >>> my_plot()  # When not specified, use default backend
        Using vispy backend with args = (), kwargs = {}
        >>> with dplt.use():  # No parameters = reset defaults (except the default backend)
        ...     my_plot()
        Using vispy backend with args = (), kwargs = {}
        >>> with dplt.use("plotly"):  # We can change the default backend
        ...     my_plot()  # So that now it defaults to 'matplotlib'
        Using plotly backend with args = (), kwargs = {}
        >>>
        >>> with dplt.use(
        ...     "plotly", color="black"
        ... ):  # We can also specify additional defaults
        ...     my_plot()
        Using plotly backend with args = (), kwargs = {'color': 'black'}
    """
    global DEFAULT_BACKEND, DEFAULT_KWARGS
    default_backend = DEFAULT_BACKEND
    default_kwargs = DEFAULT_KWARGS

    with BACKEND_LOCK:
        try:
            yield set_defaults(*args, **kwargs)
        finally:
            DEFAULT_BACKEND = default_backend
            DEFAULT_KWARGS = default_kwargs


class _Dispatcher(Generic[P, T]):
    registry: types.MappingProxyType[str, Callable[P, T]]

    def __call__(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T: ...
    def register(
        self,
        backend: str,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


def dispatch(fun: Callable[P, T]) -> _Dispatcher[P, T]:
    """
    Transform a function into a backend dispatcher for plot functions.

    Args:
        fun: The callable that will register future dispatch
            functions for each backend implementation.

    Return:
        A callable that can register backend implementations with ``register``.

    Notes:
        Only the functions registered with ``register`` will be called.
        The :data:`fun` argument wrapped inside :func:`dispatch` is
        only used for documentation, but never called.

    Examples:
        The following example shows how one can implement plotting
        utilities on different backends for a given plot.

        >>> import differt.plotting as dplt
        >>>
        >>> @dplt.dispatch
        ... def plot_line(vertices, color):
        ...     pass
        >>>
        >>> @plot_line.register("matplotlib")
        ... def _(vertices, color):
        ...     print("Using matplotlib backend")
        >>>
        >>> @plot_line.register("plotly")
        ... def _(vertices, color):
        ...     print("Using plotly backend")
        >>>
        >>> plot_line(
        ...     _,
        ...     _,
        ...     backend="matplotlib",
        ... )
        Using matplotlib backend
        >>>
        >>> plot_line(
        ...     _,
        ...     _,
        ...     backend="plotly",
        ... )
        Using plotly backend
        >>>
        >>> plot_line(
        ...     _,
        ...     _,
        ...     backend="vispy",
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
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
        Traceback (most recent call last):
        ValueError: Unsupported backend 'numpy', allowed values are: ...
    """
    registry: dict[str, Callable[P, T]] = {}

    def register(
        backend: str,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """
        Return a wrapper that will call the decorated function for the specified backend.

        Args:
            backend: The name of backend for which the decorated
                function will be called.

        Return:
            A wrapper to be put before the backend-specific implementation.
        """
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}', "
                f"allowed values are: {', '.join(SUPPORTED_BACKENDS)}."
            )

        def wrapper(impl: Callable[P, T]) -> Callable[P, T]:
            """Actually register the backend implementation."""

            @wraps(impl)
            def __wrapper__(*args: P.args, **kwargs: P.kwargs) -> T:  # noqa: N807
                try:
                    return impl(*args, **kwargs)
                except ImportError as e:
                    raise ImportError(
                        "An import error occurred when dispatching "
                        f"plot utility to backend '{backend}'. "
                        "Did you correctly install it?"
                    ) from e

            registry[backend] = __wrapper__

            return __wrapper__

        return wrapper

    @wraps(fun)
    def wrapper(
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """
        Call the appropriate backend implementation based on the default backend and the provided arguments.

        Args:
            args: Positional arguments passed to the correct backend implementation.
            kwargs: Keyword arguments passed to the correct backend implementation.

        Return:
            The result of the call.
        """
        # We cannot currently add keyword argument to the signature,
        # at least Pyright will not allow that,
        # see: https://github.com/microsoft/pyright/issues/5844.
        #
        # The motivation is detailed in P612:
        # https://peps.python.org/pep-0612/#concatenating-keyword-parameters.
        backend: str = kwargs.pop("backend", DEFAULT_BACKEND)  # type: ignore

        try:
            return registry[backend](*args, **kwargs)
        except KeyError:
            raise NotImplementedError(
                f"No backend implementation for '{backend}'"
            ) from None

        return wrapper

    wrapper.register = register  # type: ignore
    wrapper.registry = types.MappingProxyType(registry)  # type: ignore

    return wrapper  # type: ignore


def view_from_canvas(canvas: Canvas) -> ViewBox:
    """
    Return the view from the specified canvas.

    If the canvas does not have any view, create one and
    return it.

    This utility is used by :py:func:`process_vispy_kwargs`.

    Args:
        canvas: The canvas that draws the contents of the scene.

    Return:
        The view on which contents are displayed.
    """
    from vispy.scene.widgets.viewbox import ViewBox

    def default_view() -> ViewBox:
        view = canvas.central_widget.add_view()
        view.camera = "turntable"
        view.camera.depth_value = 1e3  # type: ignore
        return view

    return (
        next(
            (
                child
                for child in canvas.central_widget.children  # type: ignore
                if isinstance(child, ViewBox)
            ),
            None,  # type: ignore
        )
        or default_view()
    )


def process_vispy_kwargs(
    kwargs: MutableMapping[str, Any],
) -> tuple[Canvas, ViewBox]:
    """
    Process keyword arguments passed to some VisPy plotting utility.

    Args:
        kwargs: A mutable mapping of keyword arguments passed to VisPy plotting.

            .. warning::

                The keys specified below will be removed from the mapping.

    Keyword Args:
        canvas (:py:class:`SceneCanvas<vispy.scene.canvas.SceneCanvas>`):
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

    Return:
        The canvas and view used to display contents.
    """
    from vispy import scene

    for key, value in DEFAULT_KWARGS.items():
        kwargs.setdefault(key, value)

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
) -> tuple[MplFigure, Axes3D]:
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
        ax (:py:class:`Axes3D<mpl_toolkits.mplot3d.axes3d.Axes3D>`):
            The view on which contents are displayed. If not provided,
            will try to get axes from ``figure``
            (if supplied). The default axes will use a 3D projection.

    Warning:
        When supplying both ``figure`` and ``ax``, user
        must ensure that ``ax in figure.axes``
        evaluates to :py:data:`True`.

    Return:
        The figure and axes used to display contents.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    for key, value in DEFAULT_KWARGS.items():
        kwargs.setdefault(key, value)

    maybe_ax = kwargs.pop("ax", None)
    figure = (
        kwargs.pop("figure", None)
        or (maybe_ax.get_figure() if maybe_ax else None)
        or plt.figure()
    )

    def current_ax3d() -> Optional[Axes3D]:
        if len(figure.axes) > 0:
            ax = figure.gca()
            if isinstance(ax, Axes3D):
                return ax
        return None

    def new_ax3d() -> Axes3D:
        return figure.add_subplot(projection="3d")  # type: ignore

    ax = maybe_ax or current_ax3d() or new_ax3d()

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

    Return:
        The figure used to display contents.
    """
    import plotly.graph_objects as go

    for key, value in DEFAULT_KWARGS.items():
        kwargs.setdefault(key, value)

    return kwargs.pop("figure", None) or go.Figure()


@contextmanager
def reuse(**kwargs: Any) -> Iterator[Union[Canvas, MplFigure, Figure]]:
    """Create a context manager that will automatically reuse the current canvas / figure.

    Args:
        kwargs: Keywords arguments passed to
            :py:func:`set_defaults`.

    Return:
        The canvas or figure that is reused for this context.

    Examples:
        The following example show how the same figure is reused
        for multiple plots.

        .. plotly::

            >>> from differt.plotting import draw_image, reuse
            >>>
            >>> x = np.linspace(-1.0, +1.0, 100)
            >>> y = np.linspace(-4.0, +4.0, 200)
            >>> X, Y = np.meshgrid(x, y)
            >>>
            >>> with reuse(backend="plotly") as fig:  # doctest: +SKIP
            ...     for z0, w in enumerate(jnp.linspace(0, 10 * jnp.pi, 5)):
            ...         Z = np.cos(w * X) * np.sin(w * Y)
            ...         draw_image(Z, x=x, y=y, z0=z0)  # TODO: fix colorbar
            >>> fig  # doctest: +SKIP
    """
    global DEFAULT_KWARGS
    backend: Optional[str] = kwargs.pop("backend", None)

    with use(backend=backend) as b:
        try:
            if b == "vispy":
                canvas, view = process_vispy_kwargs(kwargs)
                DEFAULT_KWARGS = {"canvas": canvas, "view": view, **kwargs}
                yield canvas
            elif b == "matplotlib":
                figure, ax = process_matplotlib_kwargs(kwargs)
                DEFAULT_KWARGS = {"figure": figure, "ax": ax, **kwargs}
                yield figure
            else:
                figure = process_plotly_kwargs(kwargs)
                DEFAULT_KWARGS = {"figure": figure, **kwargs}
                yield figure
        finally:
            pass  # No need to reset anything, use(...) will do this
