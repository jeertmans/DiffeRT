from typing import Any

import jax
from beartype import beartype as typechecker
from jaxtyping import Array, ArrayLike, Float, jaxtyped

from differt.geometry import path_lengths

from ._constants import c


@jax.jit
@jaxtyped(typechecker=typechecker)
def lengths_to_delays(
    lengths: Float[Array, " *#batch"],
    speed: Float[ArrayLike, " *#batch"] = c,
) -> Float[Array, " *#batch"]:
    """
    Compute the delay, in seconds, corresponding to each length.

    Args:
        lengths: The array of lengths (in meters).
        speed: The speed (in meters per second)
            used to compute the delay. This can be
            an array of speeds. Default is the speed
            of light in vacuum.

    Returns:
        The array of path delays.

    Examples:
        The following example shows how to compute a delay from a length.

        >>> from differt.em import c
        >>> from differt.em import (
        ...     lengths_to_delays,
        ... )
        >>>
        >>> lengths = jnp.array([1.0, 2.0, 4.0])
        >>> lengths_to_delays(lengths) * c
        Array([1., 2., 4.], dtype=float32)
        >>> lengths_to_delays(lengths, speed=2.0)
        Array([0.5, 1. , 2. ], dtype=float32)
    """
    return lengths / speed


@jax.jit
@jaxtyped(typechecker=typechecker)
def path_delays(
    paths: Float[Array, "*batch path_length 3"],
    **kwargs: Any,
) -> Float[Array, " *batch"]:
    """
    Compute the path delay, in seconds, of each path.

    Each path is exactly made of ``path_length`` vertices.

    Args:
        paths: The array of path vertices.
        kwargs: Keyword arguments passed to
            :func:`lengths_to_delays`.

    Returns:
        The array of path delays.

    Examples:
        The following example shows how to compute the delay of a very simple path.

        >>> from differt.em import c
        >>> from differt.em import (
        ...     path_delays,
        ... )
        >>>
        >>> path = jnp.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        >>> path_delays(path) * c
        Array(1., dtype=float32)
        >>> path_delays(path, speed=2.0)
        Array(0.5, dtype=float32)
    """
    lengths = path_lengths(paths)

    return lengths_to_delays(lengths, **kwargs)
