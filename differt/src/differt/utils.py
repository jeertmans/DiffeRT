"""General purpose utilities."""

from collections.abc import Callable
from functools import partial
from typing import Any, Concatenate, ParamSpec

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, ArrayLike, Float, Num, PRNGKeyArray, Shaped


@eqx.filter_jit
def dot(
    u: Shaped[ArrayLike, "*#batch n"],
    v: Shaped[ArrayLike, "*#batch n"] | None = None,
    keepdims: bool = False,
) -> Shaped[Array, "*batch "] | Shaped[Array, "*batch 1"]:
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


# Redefined here, because chex uses deprecated type hints
# TODO: fixme when google/chex#361 is resolved.
# TODO: changme because beartype>=0.20 complains that it cannot import ArrayTree,
# and I don't see how to fix it.
OptState = Any
# TODO: fixme when Python >= 3.11
P = ParamSpec("P")


@eqx.filter_jit
def minimize(
    fun: Callable[Concatenate[Num[Array, " n"], P], Num[Array, " "]],
    x0: Num[ArrayLike, "*batch n"],
    args: tuple[Any, ...] = (),
    steps: int = 1000,
    optimizer: optax.GradientTransformation | None = None,
) -> tuple[Num[Array, "*batch n"], Num[Array, " *batch"]]:
    """
    Minimize a scalar function of one or more variables.

    The minimization is achieved by computing the
    gradient of the objective function, and performing
    a fixed (i.e., ``step``) number of iterations.

    Args:
        fun: The objective function to be minimized.
        x0: The initial guess.
        args: Positional arguments passed to ``fun``.

            Those arguments are also expected have
            batch dimensions similar to ``x0``.

            .. note::

                If your function has static arguments,
                please wrap the function with :func:`functools.partial`:

                .. code-block:: python

                    fun_p = partial(fun, static_arg=static_value)

                If your function has keyword-only
                arguments, create a wrapper function that
                maps positional arguments to keyword only arguments:

                .. code-block:: python

                    fun_p = lambda x, kw_only_value: fun(x, kw_only_arg=kw_only_value)

        steps: The number of steps to perform.
        optimizer: The optimizer to use. If not provided,
            uses :func:`optax.adam` with a learning rate of ``0.1``.

    Returns:
        The solution array and the corresponding loss.

    Examples:
        The following example shows how to minimize a basic function.

        >>> from differt.utils import minimize
        >>> import chex
        >>>
        >>> def f(x, offset=1.0):
        ...     x = x - offset
        ...     return jnp.dot(x, x)
        >>>
        >>> x, y = minimize(f, jnp.zeros(10))
        >>> chex.assert_trees_all_close(x, jnp.ones(10), rtol=1e-2)
        >>> chex.assert_trees_all_close(y, 0.0, atol=1e-4)
        >>>
        >>> # It is also possible to pass positional arguments
        >>> x, y = minimize(f, jnp.zeros(10), args=(2.0,))
        >>> chex.assert_trees_all_close(x, 2.0 * jnp.ones(10), rtol=1e-2)
        >>> chex.assert_trees_all_close(y, 0.0, atol=1e-3)
        >>>
        >>> # You can also change the optimizer and the number of steps
        >>> import optax
        >>> optimizer = optax.noisy_sgd(learning_rate=0.003)
        >>> x, y = minimize(
        ...     f, jnp.zeros(5), args=(4.0,), steps=10000, optimizer=optimizer
        ... )
        >>> chex.assert_trees_all_close(x, 4.0 * jnp.ones(5), rtol=1e-2)
        >>> chex.assert_trees_all_close(y, 0.0, atol=1e-3)

        This example shows how you can minimize on a batch of arrays.

        >>> from differt.utils import minimize
        >>> import chex
        >>>
        >>> batch = (1, 2, 3)
        >>> n = 10
        >>> key = jax.random.key(1234)
        >>> offset = jax.random.uniform(key, (*batch, n))
        >>>
        >>> def f(x, offset, scale=2.0):
        ...     x = scale * x - offset
        ...     return jnp.sum(x * x, axis=-1)
        >>>
        >>> x0 = jnp.zeros((*batch, n))
        >>> x, y = minimize(f, x0, args=(offset,), steps=1000)
        >>> chex.assert_trees_all_close(x, offset / 2.0, rtol=1e-2)
        >>> chex.assert_trees_all_close(y, 0.0, atol=1e-4)
        >>>
        >>> # By default, arguments are expected to have batch
        >>> # dimensions like 'x0'. So, if 'x0' has more than one dimension,
        >>> # 'offset' must be an ndarray with the same shape prefix as 'x0':
        >>> offset = 10.0
        >>> x, y = minimize(
        ...     f, x0, args=(offset,), steps=1000
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        TypeError: ... Tree leaf '0' is not an ndarray (type=<class 'float'>).
        >>>
        >>> # Passing 'offset' as an ndarray instead:
        >>> x, y = minimize(
        ...     f, x0, args=(jnp.array(offset),), steps=1000
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        TypeError: ... Tree leaf '1' has a shape of length 0 (shape=()) which
        is smaller than the expected prefix of length 3 (prefix=(1, 2, 3)).
        >>>
        >>> # For static arguments, use functools.partial
        >>> from functools import partial
        >>>
        >>> fp = partial(f, offset=offset)
        >>> x, y = minimize(fp, x0, steps=1000)
        >>> chex.assert_trees_all_close(x, offset * jnp.ones_like(x0) / 2.0, rtol=1e-2)
        >>> chex.assert_trees_all_close(y, 0.0, atol=1e-2)
    """
    x0 = jnp.asarray(x0)
    if x0.ndim > 1 and args:
        chex.assert_tree_has_only_ndarrays(args, exception_type=TypeError)
        chex.assert_tree_shape_prefix(
            (x0, *args), x0.shape[:-1], exception_type=TypeError
        )

    optimizer = optimizer or optax.adam(learning_rate=0.1)

    f_and_df = jax.value_and_grad(fun)

    for _ in x0.shape[:-1]:
        f_and_df = jax.vmap(f_and_df)

    opt_state = optimizer.init(x0)

    def f(
        carry: tuple[Num[Array, "*batch n"], OptState],
        _: None,
    ) -> tuple[tuple[Num[Array, "*batch n"], OptState], Num[Array, " *batch"]]:
        x, opt_state = carry
        loss, grads = f_and_df(x, *args)
        updates, opt_state = optimizer.update(grads, opt_state)
        x = x + updates
        carry = (x, opt_state)
        return carry, loss

    (x, _), losses = jax.lax.scan(f, init=(x0, opt_state), xs=None, length=steps)

    return x, losses[-1]


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
    return jnp.where(den == 0, 0, num / den)
