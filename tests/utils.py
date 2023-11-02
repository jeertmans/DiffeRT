import inspect
from collections.abc import Sequence
from functools import wraps
from typing import Any, Callable

import jax
from chex import Array


def random_inputs(
    *arg_names: str,
    sampler: Callable[[jax.random.PRNGKey, Sequence[int]], Array] = jax.random.uniform,
    seed: int = 0,
) -> Callable[..., Any]:
    """
    Wrap a function so that specified input arguments are
    randomly generated based array shapes.

    This is useful for generating random test arrays.

    :param arg_names:
        Sequence of argument names that should be transformed into random
        arrays.
    """
    arg_names = set(arg_names)  # Repeating names is useless

    def wrapper(fun: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(fun)

        @wraps(fun)
        def _wrapper_(*args: Any, **kwargs: Any) -> Any:
            bound_args = sig.bind(*args, **kwargs)
            key = jax.random.PRNGKey(seed)
            for arg_name in arg_names:
                shape = bound_args.arguments[arg_name]
                bound_args.arguments[arg_name] = sampler(key, shape)

            return fun(*bound_args.args, **bound_args.kwargs)

        return _wrapper_

    return wrapper
