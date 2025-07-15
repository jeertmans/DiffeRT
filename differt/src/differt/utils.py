"""General purpose utilities."""

from functools import partial
from typing import (
    Literal,
    overload,
)

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Num, PRNGKeyArray, Shaped


@overload
def dot(
    u: Num[ArrayLike, "*#batch n"],
    v: Num[ArrayLike, "*#batch n"] | None = None,
    *,
    keepdims: Literal[False] = False,
) -> Num[Array, "*batch "]: ...


@overload
def dot(
    u: Num[ArrayLike, "*#batch n"],
    v: Num[ArrayLike, "*#batch n"] | None = None,
    *,
    keepdims: Literal[True],
) -> Num[Array, "*batch 1"]: ...


@eqx.filter_jit
def dot(
    u: Num[ArrayLike, "*#batch n"],
    v: Num[ArrayLike, "*#batch n"] | None = None,
    *,
    keepdims: bool = False,
) -> Num[Array, "*batch "] | Num[Array, "*batch 1"]:
    """
    Compute the dot product between two multidimensional arrays, over the last axis.

    Args:
        u: The first input array.
        v: The second input array.

            If not provided, the second input is assumed to be the first input.
        keepdims: If set to :data:`True`, the output array will have the same
            number of dimensions as the input.

    Returns:
        The array of dot products.

    Examples:
        The following example shows how the dot product works.

        >>> from differt.utils import dot
        >>>
        >>> u = jnp.arange(10).reshape(5, 2)
        >>> u
        Array([[0, 1],
               [2, 3],
               [4, 5],
               [6, 7],
               [8, 9]], dtype=int32)
        >>>
        >>> dot(u)
        Array([  1,  13,  41,  85, 145], dtype=int32)
        >>> dot(u, u)
        Array([  1,  13,  41,  85, 145], dtype=int32)
        >>> dot(u, keepdims=True)
        Array([[  1],
               [ 13],
               [ 41],
               [ 85],
               [145]], dtype=int32)
    """
    u = jnp.asarray(u)
    v = jnp.asarray(v) if v is not None else u

    return jnp.sum(u * v, axis=-1, keepdims=keepdims)


@jax.jit
def sorted_array2(array: Shaped[ArrayLike, "m n"]) -> Shaped[Array, "m n"]:
    """
    Sort a 2D array by row and (then) by column.

    Args:
        array: The input array.

    Returns:
        A sorted copy of the input array.

    Examples:
        The following example shows how the sorting works.

        >>> from differt.utils import (
        ...     sorted_array2,
        ... )
        >>>
        >>> arr = jnp.array([[4, 5], [8, 9], [0, 1], [2, 3], [6, 7]])
        >>>
        >>> sorted_array2(arr)
        Array([[0, 1],
               [2, 3],
               [4, 5],
               [6, 7],
               [8, 9]], dtype=int32)
        >>>
        >>> arr = jnp.array([
        ...     [1, 1, 1, 0, 1],
        ...     [1, 0, 1, 1, 1],
        ...     [1, 0, 0, 1, 1],
        ...     [1, 0, 0, 0, 0],
        ...     [1, 1, 0, 1, 0],
        ... ])
        >>>
        >>> sorted_array2(arr)
        Array([[1, 0, 0, 0, 0],
               [1, 0, 0, 1, 1],
               [1, 0, 1, 1, 1],
               [1, 1, 0, 1, 0],
               [1, 1, 1, 0, 1]], dtype=int32)
    """
    array = jnp.asarray(array)
    if array.size == 0:
        return array

    return array[jnp.lexsort(array.T[::-1])]


@partial(jax.jit, static_argnames=("shape",))
def sample_points_in_bounding_box(
    bounding_box: Float[ArrayLike, "2 3"],
    shape: tuple[int, ...] = (),
    *,
    key: PRNGKeyArray,
) -> Float[Array, "*shape 3"]:
    """
    Sample point(s) in a 3D bounding box.

    Args:
        bounding_box: The bounding box (min. and max. coordinates).
        shape: The sample shape.
        key: The :func:`jax.random.PRNGKey` to be used.

    Returns:
        An array of points randomly sampled.
    """
    bounding_box = jnp.asarray(bounding_box)
    amin = bounding_box[0, :]
    amax = bounding_box[1, :]
    scale = amax - amin

    r = jax.random.uniform(key, shape=(*shape, 3))

    return r * scale + amin


@jax.jit
def safe_divide(
    num: Num[ArrayLike, " *#batch"], den: Num[ArrayLike, " *#batch"]
) -> Num[Array, " *batch"]:
    """
    Compute the elementwise division, but returns 0 when ``den`` is zero.

    Args:
        num: The numerator.
        den: The denominator.

    Returns:
        The result of ``num / dev``, except that division by zero returns 0.

    Examples:
        The following examples shows how division by zero is handled.

        >>> from differt.utils import safe_divide
        >>>
        >>> x = jnp.array([1, 2, 3, 4, 5])
        >>> y = jnp.array([0, 1, 2, 0, 2])
        >>> safe_divide(x, y)
        Array([0. , 2. , 1.5, 0. , 2.5], dtype=float32)
    """
    # TODO: add :python: rst role for x / y in docs
    num = jnp.asarray(num)
    den = jnp.asarray(den)
    shape = jnp.broadcast_shapes(num.shape, den.shape)
    dtype = jnp.result_type(num, den)
    zero_div = den == 0.0
    den = jnp.where(zero_div, jnp.ones_like(den), den)
    return jnp.where(zero_div, jnp.zeros(shape, dtype=dtype), num / den)


@partial(jax.jit, inline=True)
def smoothing_function(
    x: Float[ArrayLike, " *#batch"],
    /,
    smoothing_factor: Float[ArrayLike, " *#batch"] = 1.0,
) -> Float[ArrayLike, " *batch"]:
    r"""
    Return a smoothed approximation of the Heaviside step function.

    This function is used internally for smoothing-out discontinuities
    in Ray Tracing, see :ref:`smoothing`.

    Args:
        x: The input array.
        smoothing_factor: The slope---or scaling---parameter, also noted :math:`\alpha`.

    Returns:
        The output of the smoothing function.
    """
    return jax.nn.sigmoid(x * smoothing_factor)
