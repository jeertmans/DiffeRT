from functools import partial
import jax
import jax.numpy as jnp
from typing import Union, Optional, Any, Callable
from jaxtyping import Array, Bool, Float, jaxtyped, Int

ArrayLikeFloat = Union[Float[Array, " *batch"], float]
ScalarFloat = Union[Float[Array, " "], float]
ScalarInt = Union[Int[Array, " "], int]
Truthy = Union[Bool[Array, " *batch"], Float[Array, " *batch"]]
"""An array of truthy values, either booleans or floats between 0 and 1."""
DEFAULT_ALPHA = 1.0

@partial(jax.jit, inline=True)
def sigmoid(x: ArrayLikeFloat, alpha: ScalarFloat) -> Float[Array, " *batch"]:
    r"""
    Element-wise sigmoid, parametrized with ``alpha``.

    .. math::
        \text{sigmoid}(x;\alpha) = \frac{1}{1 + e^{-\alpha x}},

    where :math:`\alpha` (``alpha``) is a slope parameter.

    See :func:`jax.nn.sigmoid` for more details.

    :param x: The input array.
    :param alpha: The slope parameter.
    :return: The corresponding values.
    """
    return jax.nn.sigmoid(alpha * x)


@partial(jax.jit, inline=True)
def hard_sigmoid(x: ArrayLikeFloat, alpha: ScalarFloat) -> Float[Array, " *batch"]:
    r"""
    Element-wise sigmoid, parametrized with ``alpha``.

    .. math::
        \text{hard_sigmoid}(x;\alpha) = \frac{\text{relu6}(\alpha x + 3)}{6},

    where :math:`\alpha` (``alpha``) is a slope parameter.

    See :func:`jax.nn.hard_sigmoid` and :func:`jax.nn.relu6` for more details.

    :param x: The input array.
    :param alpha: The slope parameter.
    :return: The corresponding values.
    """
    return jax.nn.hard_sigmoid(alpha * x)

@partial(jax.jit, inline=True)
def my_approx(x: ArrayLikeFloat, alpha: ScalarFloat) -> Float[Array, " *batch"]:
    return sigmoid(x, alpha)

@partial(jax.jit, inline=True, static_argnames=("function",))
@jaxtyped(typechecker=None)
def activation(
    x: ArrayLikeFloat,
    alpha: ScalarFloat = DEFAULT_ALPHA,
    function: Callable[
        [ArrayLikeFloat, ScalarFloat],
        Float[Array, " *batch"],
    ] = sigmoid,
) -> Float[Array, " *batch"]:
    r"""
    Element-wise function for approximating a discrete transition between 0 and 1, with a smoothed transition centered at :python:`x = 0.0`.

    Depending on the ``function`` argument, the activation function has the
    different definition.

    Two basic activation functions are provided: :func:`sigmoid` and :func:`hard_sigmoid`.
    If needed, you can implement your own activation function and pass it as an argument,
    granted that it satisfies the properties defined in the related paper.

    :param x: The input array.
    :param alpha: The slope parameter.
    :return: The corresponding values.

    :EXAMPLES:

    .. plot::
        :include-source: true

        import matplotlib.pyplot as plt
        import numpy as np
        from differt2d.logic import activation, hard_sigmoid, sigmoid
        from jax import grad, vmap

        x = np.linspace(-5, +5, 200)

        _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=[6.4, 8])

        for name, function in [("sigmoid", sigmoid), ("hard_sigmoid", hard_sigmoid)]:
            def f(x):
                return activation(x, alpha=1.5, function=function)

            y = f(x)
            dydx = vmap(grad(f))(x)
            _ = ax1.plot(x, y, "--", label=f"{name}")
            _ = ax2.plot(x, dydx, "-", label=f"{name}")

        ax2.set_xlabel("$x$")
        ax1.set_ylabel("$f(x)$")
        ax2.set_ylabel(r"$\frac{\partial f(x)}{\partial x}$")
        plt.legend()
        plt.tight_layout()
        plt.show()  # doctest: +SKIP
    """
    return function(x, alpha)

@partial(jax.jit, inline=True, static_argnames=("approx",))
def logical_or(
    x: Union[Truthy, float, bool],
    y: Union[Truthy, float, bool],
    approx: Optional[bool] = None,
) -> Truthy:
    """
    Element-wise logical :python:`x or y`.

    Calls :func:`jax.numpy.maximum` if approximation is enabled,
    :func:`jax.numpy.logical_or` otherwise.

    :param x: The first input array.
    :param y: The second input array.
    :param approx: Whether approximation is enabled or not.
    :return: Output array, with element-wise comparison.
    """
    return jnp.maximum(x, y) if approx else jnp.logical_or(x, y)


@partial(jax.jit, inline=True, static_argnames=("approx",))
def logical_and(
    x: Union[Truthy, float, bool],
    y: Union[Truthy, float, bool],
    approx: Optional[bool] = None,
) -> Truthy:
    """
    Element-wise logical :python:`x and y`.

    Calls :func:`jax.numpy.minimum` if approximation is enabled,
    :func:`jax.numpy.logical_or` otherwise.

    :param x: The first input array.
    :param y: The second input array.
    :param approx: Whether approximation is enabled or not.
    :return: Output array, with element-wise comparison.
    """
    return jnp.minimum(x, y) if approx else jnp.logical_and(x, y)


@partial(jax.jit, inline=True, static_argnames=("approx",))
def logical_not(x: Union[Truthy, float, bool], approx: Optional[bool] = None) -> Truthy:
    """
    Element-wise logical :python:`not x`.

    Calls :func:`jax.numpy.subtract`
    (:python:`1 - x`) if approximation is enabled,
    :func:`jax.numpy.logical_or` otherwise.

    :param x: The input array.
    :param approx: Whether approximation is enabled or not.
    :return: Output array, with element-wise comparison.
    """
    return jnp.subtract(1.0, x) if approx else jnp.logical_not(x)

@partial(jax.jit, inline=True, static_argnames=("approx", "function"))
def greater(
    x: ArrayLikeFloat,
    y: ArrayLikeFloat,
    approx: Optional[bool] = None,
    **kwargs: Any,
) -> Truthy:
    """
    Element-wise logical :python:`x > y`.

    Calls :func:`jax.numpy.subtract`
    then :func:`activation` if approximation is enabled,
    :func:`jax.numpy.greater` otherwise.

    :param x: The first input array.
    :param y: The second input array.
    :param approx: Whether approximation is enabled or not.
    :param kwargs:
        Keyword arguments passed to :func:`activation`.
    :return: Output array, with element-wise comparison.
    """
    return activation(jnp.subtract(x, y), **kwargs) if approx else jnp.greater(x, y)


@partial(jax.jit, inline=True, static_argnames=("approx", "function"))
def greater_equal(
    x: ArrayLikeFloat,
    y: ArrayLikeFloat,
    approx: Optional[bool] = None,
    **kwargs: Any,
) -> Truthy:
    """
    Element-wise logical :python:`x >= y`.

    Calls :func:`jax.numpy.subtract`
    then :func:`activation` if approximation is enabled,
    :func:`jax.numpy.greater_equal` otherwise.

    :param x: The first input array.
    :param y: The second input array.
    :param approx: Whether approximation is enabled or not.
    :param kwargs:
        Keyword arguments passed to :func:`activation`.
    :return: Output array, with element-wise comparison.
    """
    return (
        activation(jnp.subtract(x, y), **kwargs) if approx else jnp.greater_equal(x, y)
    )


@partial(jax.jit, inline=True, static_argnames=("approx", "function"))
def less(
    x: ArrayLikeFloat,
    y: ArrayLikeFloat,
    approx: Optional[bool] = None,
    **kwargs: Any,
) -> Truthy:
    """
    Element-wise logical :python:`x < y`.

    Calls :func:`jax.numpy.subtract` (arguments swapped)
    then :func:`activation` if approximation is enabled,
    :func:`jax.numpy.less` otherwise.

    :param x: The first input array.
    :param y: The second input array.
    :param approx: Whether approximation is enabled or not.
    :param kwargs:
        Keyword arguments passed to :func:`activation`.
    :return: Output array, with element-wise comparison.
    """
    return activation(jnp.subtract(y, x), **kwargs) if approx else jnp.less(x, y)


@partial(jax.jit, inline=True, static_argnames=("approx", "function"))
def less_equal(
    x: ArrayLikeFloat,
    y: ArrayLikeFloat,
    approx: Optional[bool] = None,
    **kwargs: Any,
) -> Truthy:
    """
    Element-wise logical :python:`x <= y`.

    Calls :func:`jax.numpy.subtract` (arguments swapped)
    then :func:`activation` if approximation is enabled,
    :func:`jax.numpy.less_equal` otherwise.

    :param x: The first input array.
    :param y: The second input array.
    :param approx: Whether approximation is enabled or not.
    :param kwargs:
        Keyword arguments passed to :func:`activation`.
    :return: Output array, with element-wise comparison.
    """
    return activation(jnp.subtract(y, x), **kwargs) if approx else jnp.less_equal(x, y)


@partial(jax.jit, inline=True, static_argnames=("axis", "approx"))
def logical_all(
    *x: Union[Truthy, float, bool],
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    approx: Optional[bool] = False,
) -> Truthy:
    """
    Returns whether all values in ``x`` are true.

    Calls :func:`jax.numpy.min` if approximation is enabled,
    :func:`jax.numpy.all` otherwise.

    :param x: The input array, or array-like.
    :param axis: Axis or axes along which to operate.
        By default, flattened input is used.
    :param approx: Whether approximation is enabled or not.
    :return: Output array.
    """
    arr = jnp.asarray(x)
    if arr.size == 0:
        return jnp.array(True, dtype=bool) if not approx else jnp.array(1.0)
    return jnp.min(arr, axis=axis) if approx else jnp.all(arr, axis=axis)


@partial(jax.jit, inline=True, static_argnames=("axis", "approx"))
def logical_any(
    *x: Union[Truthy, float, bool],
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    approx: Optional[bool] = False,
) -> Truthy:
    """
    Returns whether any value in ``x`` is true.

    Calls :func:`jax.numpy.max` if approximation is enabled,
    :func:`jax.numpy.any` otherwise.

    :param x: The input array, or array-like.
    :param axis: Axis or axes along which to operate.
        By default, flattened input is used.
    :param approx: Whether approximation is enabled or not.
    :return: Output array.
    """
    arr = jnp.asarray(x)
    if arr.size == 0:
        return jnp.array(False, dtype=bool) if not approx else jnp.array(.0)
    return jnp.max(arr, axis=axis) if approx else jnp.any(arr, axis=axis)


@partial(jax.jit, inline=True, static_argnames=("approx",))
def is_true(
    x: Union[Truthy, float, bool],
    tol: ScalarFloat = 0.5,
    approx: Optional[bool] = None,
) -> Bool[Array, " *batch"]:
    """
    Element-wise check if a given truth value can be considered to be true.

    When using approximation,
    this function checks whether the value is close to 1.

    :param x: The input array.
    :param tol: The tolerance on how close it should be to 1.
        Only used if :code:`approx` is set to :py:data:`True`.
    :param approx: Whether approximation is enabled or not.
    :return: True array if the value is considered to be true.
    """
    return jnp.greater(x, 1.0 - tol) if approx else jnp.asarray(x)


@partial(jax.jit, inline=True, static_argnames=("approx",))
def is_false(
    x: Union[Truthy, float, bool],
    tol: ScalarFloat = 0.5,
    approx: Optional[bool] = None,
) -> Bool[Array, " *batch"]:
    """
    Element-wise check if a given truth value can be considered to be false.

    When using approximation,
    this function checks whether the value is close to 0.

    :param x: The input array.
    :param tol: The tolerance on how close it should be to 0.
        Only used if :code:`approx` is set to :py:data:`True`.
    :param approx: Whether approximation is enabled or not.
    :return: True if the value is considered to be false.
    """
    return jnp.less(x, tol) if approx else jnp.logical_not(x)


@partial(jax.jit, inline=False, static_argnames=("approx",))
def true_value(approx: Optional[bool] = None) -> Truthy:
    """
    Returns a scalar true value.

    When using approximation, this function returns 1.

    :param approx: Whether approximation is enabled or not.
    :return: A value that evaluates to true.
    """
    return jnp.array(1.0) if approx else jnp.array(True, dtype=bool)


@partial(jax.jit, inline=False, static_argnames=("approx",))
def false_value(approx: Optional[bool] = None) -> Truthy:
    """
    Returns a scalar false value.

    When using approximation, this function returns 0.

    :param approx: Whether approximation is enabled or not.
    :return: A value that evaluates to false.
    """
    return jnp.array(0.0) if approx else jnp.array(False, dtype=bool)
