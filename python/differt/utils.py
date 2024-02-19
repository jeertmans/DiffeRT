"""General purpose utilities."""
from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Num, Shaped, jaxtyped
from typeguard import typechecked as typechecker


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


@jaxtyped(typechecker=typechecker)
def minimize(
    fun: Callable[[Num[Array, "*batch n"]], Num[Array, " *batch"]],
    x0: Num[Array, "*batch n"],
    fun_args: tuple | None = None,
    fun_kwargs: dict[str, Any] | None = None,
    steps: int = 100,
    optimizer: optax.GradientTransformation | None = None,
) -> tuple[Num[Array, "*batch n"], Num[Array, " *batch"]]:
    """
    Minimize a scalar function of one or more variables.

    Args:
        fun: The objective function to be minimized.
        x0: The initial guess.
        fun_args: Positional arguments to be passed to ``fun``.
        fun_kwargs: Keyword arguments to be passed to ``fun``.
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
        >>> x, y = minimize(f, jnp.zeros(10), fun_args=(2.0,))
        >>> chex.assert_trees_all_close(x, 2.0 * jnp.ones(10), rtol=1e-2)
        >>> chex.assert_trees_all_close(y, 0.0, atol=1e-3)
        >>>
        >>> # Or even keyword arguments
        >>> x, y = minimize(f, jnp.zeros(10), fun_kwargs=dict(offset=3.0))
        >>> chex.assert_trees_all_close(x, 3.0 * jnp.ones(10), rtol=1e-2)
        >>> chex.assert_trees_all_close(y, 0.0, atol=1e-2)
    """
    fun_args = fun_args if fun_args else tuple()
    fun_kwargs = fun_kwargs if fun_kwargs else {}
    optimizer = optimizer if optimizer else optax.adam(learning_rate=0.1)

    f_and_df = jax.value_and_grad(fun)
    opt_state = optimizer.init(x0)

    # Cannot type check because jaxtyping fails with optax.OptState
    #  @jaxtyped(typechecker=typechecker)
    def f(
        carry: tuple[Num[Array, "*batch n"], optax.OptState], _: None
    ) -> tuple[tuple[Num[Array, "*batch n"], optax.OptState], Num[Array, " *batch"]]:
        x, opt_state = carry
        loss, grads = f_and_df(x, *fun_args, **fun_kwargs)
        updates, opt_state = optimizer.update(grads, opt_state)
        x = x + updates
        carry = (x, opt_state)
        return carry, loss

    (x, _), losses = jax.lax.scan(f, init=(x0, opt_state), xs=None, length=steps)
    return x, losses[-1]
