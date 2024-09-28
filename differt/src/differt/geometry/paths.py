"""Ray paths utilities."""

from collections.abc import Callable, Iterator
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype as typechecker
from jaxtyping import Array, ArrayLike, Bool, Float, Int, jaxtyped

from differt.plotting import draw_paths


@jaxtyped(typechecker=typechecker)
class Paths(eqx.Module):
    """
    A convenient wrapper class around path vertices and object indices.

    This class can hold arbitrary many paths, but they must share the same
    length, i.e., the same number of vertices per path.
    """

    vertices: Float[Array, "*batch path_length 3"] = eqx.field(converter=jnp.asarray)
    """The array of path vertices."""
    objects: Int[Array, "*batch path_length"] = eqx.field(converter=jnp.asarray)
    """The array of object indices.

    To every path vertex corresponds one object (e.g., a triangle).
    A placeholder value of ``-1`` can be used in specific cases,
    like for transmitter and receiver positions.
    """
    mask: Bool[Array, " *batch"] | None = eqx.field(
        converter=lambda x: jnp.asarray(x) if x is not None else None, default=None
    )
    """An optional mask to indicate which paths are valid and should be used.

    The mask is kept separately to :attr:`vertices` so that we can keep information
    batch ``*batch`` dimensions, which would not be possible if we were to directly
    store valid paths.
    """

    @property
    @jaxtyped(typechecker=typechecker)
    def path_length(self) -> int:
        """The length (i.e., number of vertices) of each individual path."""
        return self.objects.shape[-1]

    @property
    @jax.jit
    @jaxtyped(typechecker=typechecker)
    def num_valid_paths(self) -> Int[ArrayLike, ""]:
        """The number of paths kept by :attr:`mask`.

        If :attr:`mask` is not :data:`None`, then the output value can be traced by JAX.
        """
        if self.mask is not None:
            return self.mask.sum()
        return self.objects[..., 0].size

    @property
    @jaxtyped(typechecker=typechecker)
    def masked_vertices(
        self,
    ) -> Float[Array, "{self.num_valid_paths} {self.path_length} 3"]:
        """The array of masked vertices, with batched dimensions flattened into one.

        If :attr:`mask` is :data:`None`, then the returned array is simply
        :attr:`vertices` with the batch dimensions flattened.
        """
        vertices = self.vertices.reshape((-1, self.path_length, 3))
        if self.mask is not None:
            mask = self.mask.reshape(-1)
            return vertices[mask, ...]
        return vertices

    @property
    @jaxtyped(typechecker=typechecker)
    def masked_objects(
        self,
    ) -> Int[Array, "{self.num_valid_paths} {self.path_length}"]:
        """The array of masked objects, with batched dimensions flattened into one.

        Similar to :attr:`masked_vertices`, but for :data:`objects`.
        """
        objects = self.objects.reshape((-1, self.path_length))
        if self.mask is not None:
            mask = self.mask.reshape(-1)
            return objects[mask, ...]
        return objects

    @jax.jit
    @jaxtyped(typechecker=typechecker)
    def group_by_objects(self) -> Int[Array, " *batch"]:
        """
        Return an array of unique object groups.

        This function is useful to group paths that
        undergo the same types of interactions.

        Returns:
            An array of group indices.

        Examples:
            The following shows how one can group
            paths by object groups. There are two different objects,
            denoted by indices ``0`` and ``1``, and each path is made
            of three vertices.

            >>> from differt.geometry.paths import Paths
            >>>
            >>> key = jax.random.PRNGKey(1234)
            >>> key_v, key_o = jax.random.split(key, 2)
            >>> *batch, path_length = (2, 6, 3)
            >>> vertices = jax.random.uniform(key_v, (*batch, path_length, 3))
            >>> objects = jax.random.randint(key_o, (*batch, path_length), 0, 2)
            >>> objects
            Array([[[1, 0, 0],
                    [1, 0, 0],
                    [0, 1, 1],
                    [1, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1]],
                   [[1, 0, 1],
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 1, 0],
                    [0, 0, 1],
                    [0, 1, 1]]], dtype=int32)
            >>> paths = Paths(vertices, objects)
            >>> groups = paths.group_by_objects()
            >>> groups
            Array([[4, 4, 3, 5, 2, 3],
                   [5, 0, 0, 6, 1, 3]], dtype=int32)
        """
        *batch, path_length = self.objects.shape

        objects = self.objects.reshape((-1, path_length))
        inverse = jnp.unique(
            objects, axis=0, size=objects.shape[0], return_inverse=True
        )[1]

        return inverse.reshape(batch)

    def __iter__(self) -> Iterator["Paths"]:
        """Return an iterator over masked paths.

        Each item of the iterator is itself an instance :class:`Paths`,
        so you can still benefit from convenient methods like :meth:`plot`.

        Yields:
            Masked paths, one at a time.
        """
        for vertices, objects in zip(
            self.masked_vertices, self.masked_objects, strict=False
        ):
            yield Paths(vertices=vertices, objects=objects, mask=None)

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def reduce(
        self, fun: Callable[[Float[Array, "*batch n"]], Float[Array, " *batch"]]
    ) -> Float[Array, " "]:
        """Apply a function on all paths and accumulate the result into a scalar value.

        Args:
            fun: The function to be called on all path vertices.

        Returns:
            The sum of the results, with contributions from
            invalid paths that are set to zero.
        """
        return jnp.sum(fun(self.vertices), where=self.mask)

    def plot(self, **kwargs: Any) -> Any:
        """
        Plot the (masked) paths on a 3D scene.

        Args:
            kwargs: Keyword arguments passed to
                :func:`draw_paths<differt.plotting.draw_paths>`.

        Returns:
            The resulting plot output.
        """
        return draw_paths(np.asarray(self.masked_vertices), **kwargs)
