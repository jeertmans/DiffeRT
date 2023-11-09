from __future__ import annotations


from typing import Any, Callable, TYPE_CHECKING
from functools import wraps

if TYPE_CHECKING:
    from typing import ParamSpec

SUPPORTED_BACKEND = ("matplotlib", "open3d", "plotly")

class UnsupportedBackendError(ValueError):
    def __init__(self, backend: str) -> None:
        super().__init__(f"Unsupported backend '{backend}', allowed values are: {', '.join(SUPPORTED_BACKEND)}.")


def dispatch(fun: Callable[..., Any]) -> Callable[..., Any]:
    """
    Transform a function into a backend dispatcher plot function.

    Examples:

        The following example shows how one can implement plotting
        utilities on different backends for a given class.

        >>> from differt import plotting
        >>>
        >>> @plotting.dispatch
        >>> def plot_line(vertices, color):
        >>>     pass
        >>>
        >>> @plot_line.register("matplotlib")
        >>> def _(vertices, color):
        >>>     plotting.matplotlib.pyplot.plot(*vertices, color=color)
        >>>
        >>> @plot_line.register("plotly")
        >>> def _(vertices, color):
        >>>     plotting.plot(*vertices, color=color)
    """
    registry = {}

    def register(backend: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a new implemenation."""
        if backend not in SUPPORTED_BACKEND:
            raise UnsupportedBackendError(backend)

        def wrapper(impl: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(impl)
            def __wrapper__(*args: Any, **kwargs: Any) -> Any:
                return impl(*args, **kwargs)

            registry[backend] = impl

    def dispatch(backend: str) -> Callable:
        pass

    @wraps(fun)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return fun(*args, **kwargs)

    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = registry

    return wrapper


def plot_method():
    pass

def register_backend():
    pass
