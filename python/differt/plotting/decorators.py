from __future__ import annotations

from functools import wraps
from typing import Any, Callable

SUPPORTED_BACKEND = ("matplotlib", "plotly", "vispy")


def dispatch(
    fun_or_none: Callable[..., Any] | None, default_backend: str = "plotly"
) -> Callable[..., Any]:
    """
    Transform a function into a backend dispatcher for plot functions.

    Examples:
        The following example shows how one can implement plotting
        utilities on different backends for a given class.

        >>> from differt import plotting
        >>>
        >>> @plotting.dispatch
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
        ValueError: No backend implementation for 'vispy'
        >>>
        >>> @plot_line.register("numpy")  # doctest: +IGNORE_EXCEPTION_DETAIL
        ... def _(vertices, color):
        ...     pass
        ...
        Traceback (most recent call last):
        ValueError: Unsupported backend 'numpy', allowed values are: ...
    """

    def _wrapper_(fun: Callable[..., Any]) -> Callable[Callable[..., Any]]:
        registry = {}

        def register(
            backend: str,
        ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            """Register a new implemenation."""
            if backend not in SUPPORTED_BACKEND:
                raise ValueError(
                    f"Unsupported backend '{backend}', allowed values are: {', '.join(SUPPORTED_BACKEND)}."
                )

            def wrapper(impl: Callable[..., Any]) -> Callable[..., Any]:
                @wraps(impl)
                def __wrapper__(*args: Any, **kwargs: Any) -> Any:
                    return impl(*args, **kwargs)

                registry[backend] = impl

            return wrapper

        def dispatch(backend: str) -> Callable[..., Any]:
            try:
                return registry[backend]
            except KeyError:
                raise ValueError(f"No backend implementation for '{backend}'") from None

        @wraps(fun)
        def main_wrapper(*args: Any, backend: str | None = None, **kwargs: Any) -> Any:
            if backend is None:
                backend = default_backend

            return dispatch(backend)(*args, **kwargs)

        main_wrapper.register = register
        main_wrapper.dispatch = dispatch
        main_wrapper.registry = registry

        return main_wrapper

    if fun_or_none:
        return _wrapper_(fun_or_none)

    return _wrapper_


def plot_method():
    pass


def register_backend():
    pass
