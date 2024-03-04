"""General purpose utilities."""

import sys
from collections.abc import Iterable, Mapping
from functools import partial
from typing import Any, Callable, Optional, Union

import chex
import jax
import jax.numpy as jnp
import optax
from beartype import beartype as typechecker
from jaxtyping import Array, Num, Shaped, jaxtyped

if sys.version_info >= (3, 11):
    from typing import TypeVarTuple, Unpack
else:
    from typing_extensions import TypeVarTuple, Unpack

# Redefined here, because chex uses deprecated type hints
OptState = Union[chex.Array, Iterable["OptState"], Mapping[Any, "OptState"]]
Ts = TypeVarTuple("Ts")


@jax.jit
@jaxtyped(typechecker=typechecker)
def sorted_array2(array: Shaped[Array, "m n"]) -> Shaped[Array, "m n"]:
    """
    Sort a 2D array by row and (then) by column.

    Args:
        array: The input array.

    Return:
        A sorted copy of the input array.

    Examples:
        The following example shows how the sorting works.

        >>> from differt.utils import (
        ...     sorted_array2,
        ... )
        >>>
        >>> arr = jnp.arange(10).reshape(5, 2)
        >>> key = jax.random.PRNGKey(1234)
        >>> (
        ...     key1,
        ...     key2,
        ... ) = jax.random.split(key, 2)
        >>> arr = jax.random.permutation(key1, arr)
        >>> arr
        Array([[4, 5],
               [8, 9],
               [0, 1],
               [2, 3],
               [6, 7]], dtype=int32)
        >>>
        >>> sorted_array2(arr)
        Array([[0, 1],
               [2, 3],
               [4, 5],
               [6, 7],
               [8, 9]], dtype=int32)
        >>>
        >>> arr = jax.random.randint(
        ...     key2,
        ...     (5, 5),
        ...     0,
        ...     2,
        ... )
        >>> arr
        Array([[1, 1, 1, 0, 1],
               [1, 0, 1, 1, 1],
               [1, 0, 0, 1, 1],
               [1, 0, 0, 0, 0],
               [1, 1, 0, 1, 0]], dtype=int32)
        >>>
        >>> sorted_array2(arr)
        Array([[1, 0, 0, 0, 0],
               [1, 0, 0, 1, 1],
               [1, 0, 1, 1, 1],
               [1, 1, 0, 1, 0],
               [1, 1, 1, 0, 1]], dtype=int32)



    """
    if array.size == 0:
        return array

    return array[jnp.lexsort(array.T[::-1])]  # type: ignore


# Beartype does not support TypeVarTuple at the moment
@partial(jax.jit, static_argnames=("fun", "steps", "optimizer"))
@jaxtyped(typechecker=None)
def minimize(
    fun: Callable[[Num[Array, "*batch n"], *Ts], Num[Array, " *batch"]],
    x0: Num[Array, "*batch n"],
    args: tuple[Unpack[Ts]] = (),
    steps: int = 1000,
    optimizer: Optional[optax.GradientTransformation] = None,
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

            .. note::

                Those argument are also expected have
                batch dimensions similar to ``x0``.

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

    Return:
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
        >>> x, y = minimize(f, jnp.zeros(5), args=(4.0,), steps=10000, optimizer=optimizer)
        >>> chex.assert_trees_all_close(x, 4.0 * jnp.ones(5), rtol=1e-2)
        >>> chex.assert_trees_all_close(y, 0.0, atol=1e-3)

        This example shows how you can minimize on a batch of arrays.
        The signature of the objective function is ``(*batch, n) -> (*batch)``,
        where each batch is minimized independently.

        >>> from differt.utils import minimize
        >>> import chex
        >>>
        >>> batch = (1, 2, 3)
        >>> n = 10
        >>> key = jax.random.PRNGKey(1234)
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
        >>> # dimensions like `x0`, so `offset` cannot be a static
        >>> # value (i.e., float):
        >>> offset = 10.0
        >>> x, y = minimize(
        ...     f, x0, args=(offset,), steps=1000
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: vmap was requested to map its arguments along axis 0, ...
        >>>
        >>> # For static arguments, use functools.partial
        >>> from functools import partial
        >>>
        >>> fp = partial(f, offset=offset)
        >>> x, y = minimize(fp, x0, steps=1000)
        >>> chex.assert_trees_all_close(x, offset * jnp.ones_like(x0) / 2.0, rtol=1e-2)
        >>> chex.assert_trees_all_close(y, 0.0, atol=1e-2)
    """
    optimizer = optimizer if optimizer else optax.adam(learning_rate=0.1)

    f_and_df = jax.value_and_grad(fun)

    for _ in x0.shape[:-1]:
        f_and_df = jax.vmap(f_and_df)

    opt_state = optimizer.init(x0)

    @jaxtyped(typechecker=typechecker)
    def f(
        carry: tuple[Num[Array, "*batch n"], OptState], _: None
    ) -> tuple[tuple[Num[Array, "*batch n"], OptState], Num[Array, " *batch"]]:
        x, opt_state = carry
        loss, grads = f_and_df(x, *args)
        updates, opt_state = optimizer.update(grads, opt_state)
        x = x + updates
        carry = (x, opt_state)
        return carry, loss

    (x, _), losses = jax.lax.scan(f, init=(x0, opt_state), xs=None, length=steps)

    return x, losses[-1]
