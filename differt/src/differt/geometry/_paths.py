import sys
import warnings
from collections.abc import Callable, Iterator
from dataclasses import KW_ONLY
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Num, Shaped, jaxtyped

from differt.plotting import PlotOutput, draw_paths, reuse

if sys.version_info >= (3, 11):  # pragma: no cover
    from typing import Self
else:
    from typing_extensions import Self


@jax.jit
@jaxtyped(typechecker=typechecker)
def _cell_ids(
    array: Shaped[Array, "batch n"],
) -> Int[Array, " batch"]:  # pragram: no cover
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
def merge_cell_ids(
    cell_ids_a: Int[Array, " *batch"],
    cell_ids_b: Int[Array, " *batch"],
) -> Int[Array, " *batch"]:
    """
    Merge two arrays of cell indices as returned by :meth:`Paths.multipath_cells`.

    Let the returned array be ``cell_ids``,
    then ``cell_ids[i] == cell_ids[j]`` for all ``i``,
    ``j`` indices if
    ``(groups_a[i], groups_b[i]) == (groups_a[j], groups_b[j])``,
    granted that arrays have been reshaped to uni-dimensional
    arrays. Of course, this method handles multiple dimensions
    and will reshape the output array to match initial shape.

    For an actual application example, see :ref:`multipath_lifetime_map`.

    Warning:
        The indices used in the returned array have nothing to
        do with the ones used in individual arrays.

    Args:
        cell_ids_a: The first array of cell indices.
        cell_ids_b: The second array of cell indices.

    Returns:
        The new array group indices.
    """
    batch = cell_ids_a.shape
    return _cell_ids(
        jnp.stack((cell_ids_a, cell_ids_b), axis=-1).reshape(-1, 2),
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
    @jaxtyped(typechecker=typechecker)
    def order(self) -> int:
        """The length (i.e., number of vertices) of each individual path, excluding start and end vertices."""
        return self.objects.shape[-1] - 2

    @property
    @jax.jit
    @jaxtyped(typechecker=typechecker)
    def num_valid_paths(self) -> Int[ArrayLike, " "]:
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
    def multipath_cells(
        self,
        axis: int = -1,
    ) -> Int[Array, " *partial_batch"]:
        """
        Return an array of same multipath cell indices.

        Let the returned array be ``cell_ids``,
        then ``cell_ids[i] == cell_ids[j]`` for all ``i``,
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
                :meth:`TriangleScene.compute_paths<differt.scene.TriangleScene.compute_paths>`.

        Returns:
            The array of group indices.

        Raises:
            ValueError: If :attr:`mask` is None.
        """
        if self.mask is None:
            msg = "Cannot create multiplath cells from non-existing mask!"
            raise ValueError(msg)

        mask = jnp.moveaxis(self.mask, axis, -1)
        *partial_batch, last_axis = mask.shape

        return _cell_ids(mask.reshape(-1, last_axis)).reshape(partial_batch)

    @jax.jit
    @jaxtyped(typechecker=typechecker)
    def group_by_objects(self) -> Int[Array, " *batch"]:
        """
        Return an array of unique object groups.

        This function is useful to group paths that
        undergo the same types of interactions.

        Internally, it uses the same logic as
        :meth:`multipath_cells`, but applied to object indices
        rather than on mask.

        Returns:
            An array of group indices.

        Examples:
            The following shows how one can group
            paths by object groups. There are two different objects,
            denoted by indices ``0`` and ``1``, and each path is made
            of three vertices.

            >>> from differt.geometry import Paths
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
        return _cell_ids(objects).reshape(batch)

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
        self, fun: Callable[[Num[Array, "*batch path_length 3"]], Num[Array, " *batch"]]
    ) -> Num[Array, " "]:
        """Apply a function on all path vertices and accumulate the result into a scalar value.

        Args:
            fun: The function to be called on all path vertices.

        Returns:
            The sum of the results, with contributions from
            invalid paths that are set to zero.
        """
        return jnp.sum(fun(self.vertices), where=self.mask)

    def plot(self, **kwargs: Any) -> PlotOutput:
        """
        Plot the (masked) paths on a 3D scene.

        Args:
            kwargs: Keyword arguments passed to
                :func:`draw_paths<differt.plotting.draw_paths>`.

        Returns:
            The resulting plot output.
        """
        return draw_paths(self.masked_vertices, **kwargs)


@jaxtyped(typechecker=typechecker)
class SBRPaths(Paths):
    """
    Paths method generated with shooting-and-bouncing method.

    Like :class:`Paths`, but holds information of lower-order
    paths too.

    E.g., second-order paths also contain information for line-of-sight (``order = 0``)
    and first-order paths.

    Warning:
        The ``mask`` argument is ignored as it will automatically
        by overwritten with the last array in :attr:`masks`.
    """

    _: KW_ONLY
    masks: Bool[Array, " *batch path_length-1"] = eqx.field(converter=jnp.asarray)
    """An array of masks.

    Extends :attr:`mask`, with one mask for each path order.
    """

    def __post_init__(self) -> None:
        if self.mask is not None:
            msg = (
                "Setting 'mask' argument is ignored for this class, "
                "as it is overwritten by 'masks' argument."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)

        self.mask = self.masks[..., -1]

    def get_paths(self, order: int) -> Paths:
        """
        Return the :class:`Paths` class instance corresponding to the given path order.

        Args:
            order: The order of the path to index.

        Returns:
            The corresponding paths class.

        Raises:
            ValueError: If the provided order is out-of-bounds.
        """
        if order < 0 or order > self.order:
            msg = (
                f"Paths order must be strictly between 0 and {self.order} (incl.), "
                f"but you provided {order}."
            )
            raise ValueError(msg)

        vertices = jnp.concatenate(
            (self.vertices[..., : order + 1, :], self.vertices[..., -1:, :]),
            axis=-2,
        )
        objects = jnp.concatenate(
            (self.objects[..., : order + 1], self.objects[..., -1:]),
            axis=-1,
        )
        return Paths(vertices=vertices, objects=objects, mask=self.masks[..., order])

    @jaxtyped(
        typechecker=None
    )  # typing.Self is (currently) not compatible with jaxtyping and beartype
    def reshape(self, *batch: int) -> Self:
        return eqx.tree_at(
            lambda p: p.masks,
            super().reshape(*batch),
            self.masks.reshape(*batch, self.masks.shape[-1]),
        )

    def plot(self, **kwargs: Any) -> PlotOutput:
        backend = kwargs.pop(
            "backend", None
        )  # TODO: check if kwargs may not cause issues
        with reuse(backend=backend) as output:
            for order in range(self.order + 1):
                self.get_paths(order).plot(**kwargs)

        return output
