# NumPy vs JAX arrays

[NumPy](https://numpy.org/)
is probably the *de facto* array library for Python, at least
for most common applications. It comes with a large set of utilities,
and many other Python modules are built on top, like
[SciPy](https://scipy.org/).

However, it lacks one key component we need for differentiability:
automatic differentiation (autodiff). And this is where
[JAX](https://github.com/jax-ml/jax) comes into play. While providing
an almost identical syntax to NumPy, JAX offers autodiff everywhere
JAX arrays are used.

Additionally, JAX scales very well on modern architectures, like GPUs and TPUs,
and provides just-in-time compilation to optimize your code.

## Where JAX arrays are used

As a result of the aforementioned pros, JAX arrays are used in the vast
majority of the codebase, both as input and as output types.

JAX arrays use the following type annotations:
`Dtype[Array, 'Shape']` where `Dtype` refers
to the
[type of array elements](https://docs.kidger.site/jaxtyping/api/array/#dtype),
and `Shape` describes
the [array shape](https://docs.kidger.site/jaxtyping/api/array/#shape).

## Where NumPy arrays are used

In some cases, using JAX arrays is just not possible.
We can identify two specific cases:

1. For plotting, we rely on third-party libraries that
   may not support JAX arrays, e.g., Vispy. As a result,
   {mod}`differt.plotting` automatically converts NumPy arrays.
2. In the Rust code, there is no way of directly creating JAX
   arrays, but works well for NumPy. Therefore, directly calling the functions
   declared with Rust code will return NumPy arrays.

Similarly, NumPy arrays use the following type annotations:
`Dtype[np.ndarray, 'Shape']`.

## From JAX to NumPy and vice-versa

Going from one array type to another is pretty simple thanks
to {func}`jnp.asarray<jax.numpy.asarray>` and {func}`np.asarray<numpy.asarray>`:

```python
>>> import jax.numpy as jnp
>>> import numpy as np
>>>
>>> jax_array = jnp.zeros(10)
>>> jax_array
Array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
>>> numpy_array = np.asarray(jax_array)
>>> numpy_array
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
>>> jax_array_back = jnp.asarray(numpy_array)
>>> jax_array_back
Array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)

```
