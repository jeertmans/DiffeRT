"""Ray paths utilities."""

import sys
from collections.abc import Callable, Iterator
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype as typechecker
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Shaped, jaxtyped

from differt.plotting import draw_paths

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


@jax.jit
@jaxtyped(typechecker=typechecker)
def _cluster_ids(array: Shaped[Array, "batch n"]) -> Int[Array, " batch"]:
    @jaxtyped(typechecker=typechecker)
    def scan_fun(
        indices: Int[Array, " batch"],
        row_and_index: tuple[Shaped[Array, " n"], Int[Array, " "]],
    ) -> tuple[Int[Array, " batch"], None]:
        row, index = row_and_index
        indices = jnp.where((array == row).all(axis=-1), index, indices)
        return indices, None

    return jax.lax.scan(
        scan_fun,
        init=jnp.empty(array.shape[0], dtype=jnp.int32),
        xs=(array, jnp.arange(array.shape[0])),
        reverse=True,
    )[0]


@jax.jit
@jaxtyped(typechecker=typechecker)
def merge_cluster_ids(
    cluster_ids_a: Int[Array, " *batch"],
    cluster_ids_b: Int[Array, " *batch"],
) -> Int[Array, " *batch"]:
    """
    Merge two arrays of cluster indices as returned by :meth:`Paths.multipath_clusters`.

    Let the returned array be ``cluster_ids``,
    then ``cluster_ids[i] == cluster_ids[j]`` for all ``i``,
    ``j`` indices if
    ``(groups_a[i], groups_b[i]) == (groups_a[j], groups_b[j])``,
    granted that arrays have been reshaped to uni-dimensional
    arrays. Of course, this method handles multiple dimensions
    and will reshape the output array to match initial shape.

    Warning:
        The indices used in the returned array have nothing to
        do with the ones used in individual arrays.

    Args:
        cluster_ids_a: The first array of cluster indices.
        cluster_ids_b: The second array of cluster indices.

    Returns:
        The new array group indices.
    """
    batch = cluster_ids_a.shape
    return _cluster_ids(
        jnp.stack((cluster_ids_a, cluster_ids_b), axis=-1).reshape(-1, 2),
    ).reshape(batch)


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

    @jaxtyped(
        typechecker=None
    )  # typing.Self is (currently) not compatible with jaxtyping and beartype
    def reshape(self, *batch: int) -> Self:
        """
        Return a copy with reshaped paths' batch dimensions to match a given shape.

        Args:
            batch: The new batch shapes.

        Returns:
            A new paths instance with specified batch dimensions.
        """
        vertices = self.vertices.reshape(*batch, self.path_length, 3)
        objects = self.objects.reshape(*batch, self.path_length)
        mask = self.mask.reshape(*batch) if self.mask is not None else None

        return eqx.tree_at(
            lambda p: (p.vertices, p.objects, p.mask),
            self,
            (vertices, objects, mask),
            is_leaf=lambda x: x is None,
        )

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

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def multipath_clusters(
        self,
        axis: int = -1,
    ) -> Int[Array, " *partial_batch"]:
        """
        Return an array of same multipath cluster indices.

        Let the returned array be ``cluster_ids``,
        then ``cluster_ids[i] == cluster_ids[j]`` for all ``i``,
        ``j`` indices if ``self.mask[i, :] == self.mask[j, :]``,
        granted that each array has been reshaped to a two-dimensional
        array and that ``axis`` is the last dimension. Of course, this
        method handles multiple dimensions and will reshape the output
        array to match initial shape, except for dimension ``axis``
        that is removed.

        The purpose of this method is to easily identify similar
        multipath structures, when a group of paths all have the
        same path candidates that are valid.

        If the different path candidates are not all on the same axis,
        e.g., as a result of masking invalid paths, then you can still
        use :meth:`group_by_objects` to identify similar paths.
        Note that :meth:`group_by_objects` will possibly return
        different indices for different transmitter / receiver pairs,
        as they have different indices. To avoid this, you should probably
        slice the :attr:`objects` array to exclude first and last objects, i.e.,
        with ``self.objects[..., 1:-1]``.

        Args:
            axis: The axis along to compare paths.

                By default, the last axis is used to match the
                ``num_path_candidates`` axis as returned by
                :meth:`TriangleScene.compute_paths<differt.scene.triangle_scene.TriangleScene.compute_paths`.

        Returns:
            The array of group indices.

        Raises:
            ValueError: If :attr:`mask` is None.
        """
        if self.mask is None:
            msg = "Cannot create multiplath clusters from non-existing mask!"
            raise ValueError(msg)

        mask = jnp.moveaxis(self.mask, axis, -1)
        *partial_batch, last_axis = mask.shape

        return _cluster_ids(mask.reshape(-1, last_axis)).reshape(partial_batch)

    @jax.jit
    @jaxtyped(typechecker=typechecker)
    def group_by_objects(self) -> Int[Array, " *batch"]:
        """
        Return an array of unique object groups.

        This function is useful to group paths that
        undergo the same types of interactions.

        Internally, it uses the same logic as
        :meth:`multipath_clusters`, but applied to object indices
        rather than on mask.

        Returns:
            An array of group indices.

        Examples:
            The following shows how one can group
            paths by object groups. There are two different objects,
            denoted by indices ``0`` and ``1``, and each path is made
            of three vertices.

            >>> from differt.geometry.paths import Paths
            >>>
            >>> key = jax.random.key(1234)
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
            Array([[ 0,  0,  2,  3,  4,  2],
                   [ 3,  7,  7,  9, 10,  2]], dtype=int32)
        """
        *batch, path_length = self.objects.shape

        objects = self.objects.reshape((-1, path_length))
        return _cluster_ids(objects).reshape(batch)

    def __iter__(self) -> Iterator[Self]:
        """Return an iterator over masked paths.

        Each item of the iterator is itself an instance :class:`Paths`,
        so you can still benefit from convenient methods like :meth:`plot`.

        Yields:
            Masked paths, one at a time.
        """
        cls = type(self)
        for vertices, objects in zip(
            self.masked_vertices, self.masked_objects, strict=False
        ):
            yield cls(vertices=vertices, objects=objects, mask=None)

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
