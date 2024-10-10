import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any

import jax
from jaxtyping import Array, PRNGKeyArray


def random_inputs(
    *arg_names: str,
    sampler: Callable[[PRNGKeyArray, tuple[int, ...]], Array] = jax.random.uniform,
    seed: int = 1234,
) -> Callable[..., Any]:
    """
    Wrap a function so that specified input arguments are
    randomly generated based array shapes.

    This is useful for generating random test arrays.

    :param arg_names:
        Sequence of argument names that should be transformed into random
        arrays.
    """

    def wrapper(fun: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(fun)

        @wraps(fun)
        def _wrapper_(*args: Any, **kwargs: Any) -> Any:
            bound_args = sig.bind(*args, **kwargs)
            keys = jax.random.split(jax.random.key(seed), len(arg_names))
            for key, arg_name in zip(keys, arg_names, strict=False):
                shape = bound_args.arguments[arg_name]
                bound_args.arguments[arg_name] = sampler(key, shape)

            return fun(*bound_args.args, **bound_args.kwargs)

        return _wrapper_

    return wrapper
