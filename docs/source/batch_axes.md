(batch-axes)=
# The `*batch` axes

As you will probably notice, many functions we provided accept
arrays with set of leading axes, referred to as `*batch`.

This notation indicates that you may provide arbitrary many dimensions
for the batch axes, and the output will preserve those batch dimensions
in the output.

For example, the array annotation `Float[Array, "*batch n 3"]`
indicates that the array must have *at least* two dimensions,
with the last one equal to 3. Additional dimensions will be
considered as batch dimensions.

For more details on array annotations,
see [NumPy vs JAX arrays](numpy_vs_jax.md#numpy-vs-jax-arrays).

## Why we provide batch axes

By design, the `*batch` axes are optional and functions
will work just fine if you do not provide any additional dimensions.

However, in Ray Tracing applications, many functions are called
repeatedly on a number of samples, e.g.,
the {func}`image_method<differt.rt.image_method.image_method>` will be
called on thousands, if not millions, of path candidates. For
every path candidate, you may also want to repeat for every pair of
transmitter and receiver locations.

Thus, allowing for arbitrary batch dimensions will help you write
code in a way that is mostly transparent to the number of *repetitions*.

E.g., the following function computes the dot product
between batch of arrays:

```python
>>> import jax
>>> import jax.numpy as jnp
>>> from jaxtyping import Array, Num
>>>
>>> def dot(
...     x: Num[Array, "*batch n"], y: Num[Array, "*batch n"]
... ) -> Num[Array, " *batch"]:
...     return jnp.sum(x * y, axis=-1)
>>>
>>> *batch, n = 40, 10, 30, 3  # batch = (40, 10, 30), n = 3
>>>
>>> x = jnp.ones((*batch, n)) * 1.0
>>> y = jnp.ones((*batch, n)) * 2.0
>>> z = dot(x, y)
>>>
>>> z.shape
(40, 10, 30)
>>> jnp.allclose(z, 1.0 * 2.0 * n)
Array(True, dtype=bool)
>>> # Of course, you can always use such functions without any *batch axes:
>>> x = jnp.array([1., 2., 3.])
>>> dot(x, x)  # 1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0
Array(14., dtype=float32)

```

That is, the resulting output will have a shape of `*batch`,
where is entry is the result of the dot produt between `n` pairs of values
from the corresponding entries in `x` and `y` input arguments.

## When batch axes are not available

If a function does not offer batch axes, there are two possibilities:

1. you can use vectorization functions, like {func}`jax.vmap`, to
   call a repeat a given function over another array axis;
2. or you think code would really benefit from having batch axes.
   In that case, we recommend opening an issue on
   [GitHub](https://github.com/jeertmans/DiffeRT).

For the latter, you can also directly suggest a patch if you know how to
implement the batch axes.
