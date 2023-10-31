import inspect
from functools import wraps
from typing import Any, Callable

import jax.numpy as jnp


def asjaxarray(*arg_names: str) -> Callable[..., Any]:
    """
    Wrap specified arguments with :func:`jax.numpy.asarray`.

    This is useful for generating test arrays with :mod:`numpy`, especially random
    ones, and to still use Jax arrays later.

    :param arg_names:
        Sequence of argument names that should be wrapped.
    """
    arg_names = tuple(set(arg_names))  # Repeating names is useless

    def wrapper(fun: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(fun)

        @wraps(fun)
        def _wrapper_(*args: Any, **kwargs: Any) -> Any:
            bound_args = sig.bind(*args, **kwargs)
            for arg_name in arg_names:
                bound_args.arguments[arg_name] = jnp.asarray(
                    bound_args.arguments[arg_name]
                )

            return fun(*bound_args.args, **bound_args.kwargs)

        return _wrapper_

    return wrapper
