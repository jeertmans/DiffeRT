"""
Useful decorators for plotting.
"""

from __future__ import annotations

import importlib

from functools import wraps
from typing import Any, Callable

CURRENT_BACKEND = None
DEFAULT_BACKEND = "vispy"
SUPPORTED_BACKENDS = ("vispy", "matplotlib", "plotly")


def use(backend: str) -> None:
    """
    Tell future plotting utilities to use this backend by default.

    Args:
        backend: The name of the backend to use.

    Raises:
        ValueError: If the backend is not supported.
        ImportError: If the backend is not installed.
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


def dispatch(
    fun: Callable[..., Any]
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
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a new implemenation."""
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}', "
                f"allowed values are: {', '.join(SUPPORTED_BACKENDS)}."
            )

        def wrapper(impl: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(impl)
            def __wrapper__(*args: Any, **kwargs: Any) -> Any:
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

    def dispatch(backend: str) -> Callable[..., Any]:
        try:
            return registry[backend]
        except KeyError:
            raise NotImplementedError(f"No backend implementation for '{backend}'") from None

    @wraps(fun)
    def main_wrapper(*args: Any, backend: str | None = None, **kwargs: Any) -> Any:
        return dispatch(backend or DEFAULT_BACKEND)(*args, **kwargs)

    main_wrapper.register = register
    main_wrapper.dispatch = dispatch
    main_wrapper.registry = registry

    return main_wrapper
