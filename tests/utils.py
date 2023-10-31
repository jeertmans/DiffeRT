from functools import wraps
import jax.numpy as jnp
from typing import Callable

import inspect


def asjaxarray(*arg_names: str) -> Callable:
    """
    Function decorator to wrap specified arguments with :func:`jax.numpy.asarray`.

    This is useful for generating test arrays with :mod:`numpy`, especially random
    ones, and to still use Jax arrays later.
    """
    arg_names = tuple(set(arg_names))  # Repeating names is useless

    def wrapper(fun: Callable):
        sig = inspect.signature(fun)

        @wraps(fun)
        def _wrapper_(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            for arg_name in arg_names:
                bound_args.arguments[arg_name] = jnp.asarray(
                    bound_args.arguments[arg_name]
                )

            return fun(*bound_args.args, **bound_args.kwargs)

        return _wrapper_

    return wrapper
