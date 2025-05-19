"""Private utility functions for differt."""

__all__ = ("asarray",)

import inspect
from typing import Any

import jax.numpy as jnp


class AsArrayMeta(type):
    def __getitem__(cls, item: Any | tuple[Any, Any]) -> "AsArray":
        if isinstance(item, tuple):
            a_type, r_type = item
        else:
            a_type = Any
            r_type = item
        return AsArray(a_type, r_type)


class AsArray(metaclass=AsArrayMeta):
    def __init__(self, a_type: Any, r_type: Any) -> None:
        sig = inspect.signature(jnp.asarray)
        parameters = list(sig.parameters.values())
        parameters[0] = parameters[0].replace(annotation=a_type)

        self.__signature__ = sig.replace(
            parameters=parameters, return_annotation=r_type
        )
        self.__annotations__ = {"a": a_type, "return": r_type}

    __call__ = lambda _self, *args, **kwargs: jnp.asarray(*args, **kwargs)  # noqa: E731


asarray = AsArray
