"""General purpose utilities."""

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Num, PRNGKeyArray


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
        The result of ``num / den``, except that division by zero returns 0.

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
