import typing
import warnings
from collections.abc import Callable, Iterator, Sequence
from itertools import zip_longest
from typing import TYPE_CHECKING, Any, overload

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Num, Shaped

from differt.plotting import PlotOutput, draw_paths, reuse

if TYPE_CHECKING or hasattr(typing, "GENERATING_DOCS"):
    from typing import Self
else:
    Self = Any  # Because runtime type checking from 'beartype' will fail when combined with 'jaxtyping'


@jax.jit
def _cell_ids(
    array: Shaped[Array, "batch n"],
) -> Int[Array, " batch"]:
    def scan_fun(
        indices: Int[Array, " batch"],
        row_and_index: tuple[Shaped[Array, " n"], Int[Array, ""]],
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
def merge_cell_ids(
    cell_ids_a: Int[ArrayLike, " *batch"],
    cell_ids_b: Int[ArrayLike, " *batch"],
) -> Int[Array, " *batch"]:
    """
    Merge two arrays of cell indices as returned by :meth:`TracedPaths.multipath_cells`.

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
        cell_ids_a: First cell index array.
        cell_ids_b: Second cell index array.

    Returns:
        Merged cell indices.
    """
    cell_ids_a = jnp.asarray(cell_ids_a)
    cell_ids_b = jnp.asarray(cell_ids_b)
    batch = cell_ids_a.shape
    return _cell_ids(
        jnp.stack((cell_ids_a, cell_ids_b), axis=-1).reshape(-1, 2),
    ).reshape(batch)


class TracedPaths(eqx.Module):
    """
    A convenient wrapper class around path vertices and object indices.

    This class can hold arbitrary many paths, but they must share the same
    length, i.e., the same number of vertices per path.
    """

    vertices: Float[Array, "*batch path_length 3"]
    """The array of path vertices."""
    objects: Int[Array, "*batch path_length"]
    """The array of object indices.

    To every path vertex corresponds one object (e.g., a triangle).
    A placeholder value of ``-1`` can be used in specific cases,
    like for transmitter and receiver positions.
    """
    mask: Bool[Array, " *batch"] | Float[Array, " *batch"]
    """A mask to indicate which paths are valid and should be used.

    The mask is kept separately to :attr:`vertices` so that we can keep information about
    batch ``*batch`` dimensions, which would not be possible if we were to directly
    store valid paths.

    If :attr:`mask` contains floating-point values, then they are interpreted as confidence
    values between 0 and 1, where values greater than or equal to :attr:`confidence_threshold`
    are considered valid.
    """
    interaction_types: Int[Array, "*batch path_length-2"]
    """An array to indicate the type of each interaction.

    Values match :attr:`InteractionType<differt.em.InteractionType>`.
    A value of ``-1`` indicates an inactive or padded interaction.
    """
    confidence_threshold: Float[ArrayLike, " "] = 0.5
    """A threshold used to decide, e.g., when plotting, whether a given path is valid or not.

    A path is considered valid if its confidence is greater than or equal to this threshold.
    Unused if :attr:`mask` is of type :class:`bool`.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """The batch shape of the paths."""
        return self.vertices.shape[:-2]

    def reshape(self, *batch: int) -> Self:
        """
        Return a new paths instance with reshaped paths' batch dimensions to match a given shape.

        Args:
            batch: New batch shape.

        Returns:
            A new paths instance with specified batch dimensions.
        """
        vertices = self.vertices.reshape(*batch, self.path_length, 3)
        resolved_batch = vertices.shape[:-2]
        objects = self.objects.reshape(*resolved_batch, self.path_length)
        mask = self.mask.reshape(*resolved_batch)
        interaction_types = self.interaction_types.reshape(
            *resolved_batch, self.path_length - 2
        )

        return eqx.tree_at(
            lambda p: (
                p.vertices,
                p.objects,
                p.mask,
                p.interaction_types,
            ),
            self,
            (vertices, objects, mask, interaction_types),
        )

    def squeeze(self, axis: int | Sequence[int] | None = None) -> Self:
        """
        Return a new paths instance by squeezing one or more axes of paths' batch dimensions.

        Args:
            axis: See :func:`jax.numpy.squeeze` for allowed values.

        Returns:
            A new paths instance with squeezed batch dimensions.

        Raises:
            ValueError: If one of the provided axes is out-of-bounds,
                or if trying to squeeze a 0-dimensional batch.
        """
        ndim = self.vertices.ndim - 2
        if axis is not None and ndim == 0:
            msg = "Cannot squeeze a 0-dimensional batch!"
            raise ValueError(msg)
        if isinstance(axis, int):
            axis = (axis,)
        if isinstance(axis, Sequence):
            axis = tuple(a + ndim if a < 0 else a for a in axis)

            if any(ax >= ndim or ax < 0 for ax in axis):
                msg = "One of the provided axes is out-of-bounds!"
                raise ValueError(msg)

        mask = self.mask.squeeze(axis)
        vertices = self.vertices.squeeze(axis)
        objects = self.objects.squeeze(axis)
        interaction_types = self.interaction_types.squeeze(axis)

        return eqx.tree_at(
            lambda p: (
                p.vertices,
                p.objects,
                p.mask,
                p.interaction_types,
            ),
            self,
            (vertices, objects, mask, interaction_types),
        )

    @eqx.filter_jit
    def mask_duplicate_objects(self, axis: int = -1) -> Self:
        """
        Return a new paths instance by masking duplicate objects along a given axis.

        E.g., when generating path candidates from a generative Machine Learning model,
        see :ref:`sampling-paths`, it is possible that the model generates the same
        path candidate multiple times. This method allows to mask these duplicates,
        while maintaining the same batch dimensions and compatibility with :func:`jax.jit`.

        Args:
            axis: The batch axis along which the unique values are computed.

                It defaults to the last axis, which is the axis where
                different path candidates are stored when generating
                paths with
                :meth:`Scene.trace_paths<differt.scene.Scene.trace_paths>`.

        Returns:
            A new paths instance with masked duplicate objects.

        Raises:
            ValueError: If the provided axis is out-of-bounds.
        """
        ndim = self.objects.ndim - 1
        batch = self.objects.shape[:-1]
        if not -ndim <= axis < ndim:
            msg = f"The provided axis {axis} is out-of-bounds for batch of dimensions {ndim}!"
            raise ValueError(msg)

        size = batch[axis]

        objects = jnp.moveaxis(self.objects, axis if axis >= 0 else axis - 1, -2)
        indices = jnp.arange(size, dtype=objects.dtype)

        def f(
            objects: Int[Array, "axis_length path_length"],
        ) -> Bool[Array, " axis_length"]:
            _, index = jnp.unique(
                objects,
                axis=0,
                size=size,
                return_index=True,
            )

            return jnp.isin(indices, index)

        for _ in range(max(ndim - 1, 0)):
            f = jax.vmap(f)

        non_duplicates = f(objects)
        non_duplicates = jnp.moveaxis(non_duplicates, -1, axis)

        return eqx.tree_at(
            lambda p: p.mask,
            self,
            self.mask * non_duplicates,
        )

    @property
    def path_length(self) -> int:
        """The length (i.e., number of vertices) of each individual path."""
        return self.objects.shape[-1]

    @property
    def order(self) -> int:
        """The length (i.e., number of vertices) of each individual path, excluding start and end vertices."""
        return self.path_length - 2

    @property
    def num_valid_paths(self) -> Int[Array, ""]:
        """The number of paths kept by :attr:`mask`.

        The output value can be traced by JAX.
        """
        if self.mask.dtype == jnp.bool_:
            return self.mask.sum()
        return (self.mask >= self.confidence_threshold).sum()

    @property
    def masked_vertices(
        self,
    ) -> Float[Array, "num_valid_paths path_length 3"]:
        """The array of masked vertices, with batched dimensions flattened into one."""
        vertices = self.vertices.reshape((-1, self.path_length, 3))
        mask = self.mask.reshape(-1)
        if mask.dtype != jnp.bool_:
            mask = mask >= self.confidence_threshold
        return vertices[mask, ...]

    @property
    def masked_objects(
        self,
    ) -> Int[Array, "num_valid_paths path_length"]:
        """The array of masked objects, with batched dimensions flattened into one.

        Similar to :attr:`masked_vertices`, but for :data:`objects`.
        """
        objects = self.objects.reshape((-1, self.path_length))
        mask = self.mask.reshape(-1)
        if mask.dtype != jnp.bool_:
            mask = mask >= self.confidence_threshold
        return objects[mask, ...]

    def masked(self) -> "TracedPaths":
        """Return a flattened version of this object that only keeps valid paths.

        The returned object has all batch dimensions flattened into one,
        keeping only the paths where :attr:`mask` is :data:`True` (or where :attr:`mask` is greater than or equal to :attr:`confidence_threshold`).

        Returns:
            A new paths instance with flattened batch dimensions and only valid paths.
        """
        paths = self.reshape(-1)

        mask = paths.mask
        if mask.dtype != jnp.bool_:
            mask = mask >= self.confidence_threshold

        return eqx.tree_at(
            lambda p: (
                p.vertices,
                p.objects,
                p.mask,
                p.interaction_types,
            ),
            paths,
            (
                paths.vertices[mask, ...],
                paths.objects[mask, ...],
                mask[mask],
                paths.interaction_types[mask, ...],
            ),
        )

    @eqx.filter_jit
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
                :meth:`Scene.trace_paths<differt.scene.Scene.trace_paths>`.

        Returns:
            The array of group indices.
        """
        mask = jnp.moveaxis(self.mask, axis, -1)
        if mask.dtype != jnp.bool_:
            mask = mask >= self.confidence_threshold
        *partial_batch, last_axis = mask.shape

        return _cell_ids(mask.reshape(-1, last_axis)).reshape(partial_batch)

    @jax.jit
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

            >>> from differt.geometry import TracedPaths
            >>>
            >>> objects = jnp.array([
            ...     [[1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1]],
            ...     [[1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0]],
            ... ])
            >>> key = jax.random.key(1234)
            >>> vertices = jax.random.uniform(key, (*objects.shape, 3))
            >>> mask = jnp.ones(objects.shape[:-1], dtype=bool)
            >>> interaction_types = jnp.zeros(
            ...     (*objects.shape[:-1], objects.shape[-1] - 2), dtype=jnp.int32
            ... )
            >>> paths = TracedPaths(
            ...     vertices, objects, mask=mask, interaction_types=interaction_types
            ... )
            >>> groups = paths.group_by_objects()
            >>> groups
            Array([[0, 1, 2, 3, 4, 4],
                   [3, 4, 1, 0, 1, 3]], dtype=int32)
        """
        *batch, path_length = self.objects.shape

        objects = self.objects.reshape((-1, path_length))
        return _cell_ids(objects).reshape(batch)

    def __iter__(self) -> Iterator["TracedPaths"]:
        """Return an iterator over masked paths.

        Each item of the iterator is itself an instance :class:`TracedPaths`,
        so you can still benefit from convenient methods like :meth:`plot`.

        Yields:
            Masked paths, one at a time.
        """
        masked = self.masked()

        for vertices, objects, interaction_types in zip_longest(
            masked.vertices,
            masked.objects,
            masked.interaction_types,
        ):
            yield TracedPaths(
                vertices=vertices,
                objects=objects,
                mask=jnp.ones((), dtype=jnp.bool_),
                confidence_threshold=masked.confidence_threshold,
                interaction_types=interaction_types,
            )

    @overload
    def reduce(
        self,
        fun: Callable[[Num[Array, "*batch path_length 3"]], Num[Array, " *batch"]],
        axis: None = ...,
    ) -> Num[Array, ""]: ...

    @overload
    def reduce(
        self,
        fun: Callable[[Num[Array, "*batch path_length 3"]], Num[Array, " *batch"]],
        axis: int | Sequence[int],
    ) -> Num[Array, " *reduced_batch"]: ...

    def reduce(
        self,
        fun: Callable[[Num[Array, "*batch path_length 3"]], Num[Array, " *batch"]],
        axis: int | Sequence[int] | None = None,
    ) -> Num[Array, ""] | Num[Array, " *reduced_batch"]:
        """Apply a function on all path vertices and accumulate the result into a scalar value (or an array if ``axis`` is provided).

        Args:
            fun: Function to apply on path vertices.
            axis: See :func:`jax.numpy.sum` for allowed values.

        Returns:
            The sum of the results, with contributions from
            invalid paths that are set to zero.
        """
        if self.mask.dtype != jnp.bool_:
            return jnp.sum(fun(self.vertices) * self.mask, axis=axis)

        return jnp.sum(fun(self.vertices), axis=axis, where=self.mask)

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


# Deprecated alias
class Paths(TracedPaths):
    """
    Deprecated alias for :class:`TracedPaths`.

    .. deprecated:: 0.10
        Use :class:`TracedPaths` instead.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "Paths is deprecated, use TracedPaths instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class LaunchedPaths(eqx.Module):
    """
    Paths method generated with ray launching methods.

    Holds information of lower-order paths too, and holds multi-order mask information.
    Not a subclass of TracedPaths.
    """

    vertices: Float[Array, "*batch path_length 3"]
    """The array of path vertices."""
    objects: Int[Array, "*batch path_length"]
    """The array of object indices.

    To every path vertex corresponds one object (e.g., a triangle).
    A placeholder value of ``-1`` can be used in specific cases,
    like for transmitter and receiver positions.
    """
    masks: Bool[Array, " *batch path_length-1"]
    """An array of masks.

    Holds one mask for each path order.
    """
    interaction_types: Int[Array, "*batch path_length-2"]
    """An array to indicate the type of each interaction.

    Values match :attr:`InteractionType<differt.em.InteractionType>`.
    A value of ``-1`` indicates an inactive or padded interaction.
    """
    confidence_threshold: Float[ArrayLike, " "] = 0.5
    """A threshold used to decide whether a given path is valid or not."""

    @property
    def shape(self) -> tuple[int, ...]:
        """The batch shape of the paths."""
        return self.vertices.shape[:-2]

    @property
    def path_length(self) -> int:
        """The length (i.e., number of vertices) of each individual path."""
        return self.objects.shape[-1]

    @property
    def order(self) -> int:
        """The length (i.e., number of vertices) of each individual path, excluding start and end vertices."""
        return self.path_length - 2

    @property
    def mask(self) -> Bool[Array, " *batch"]:
        """Alias property to the highest-order mask for backwards compatibility."""
        return self.masks[..., -1]

    def get_paths(self, order: int) -> TracedPaths:
        """
        Return the :class:`TracedPaths` class instance corresponding to the given path order.

        Args:
            order: The order of the path to index.

        Returns:
            The corresponding trace paths class.

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
        return TracedPaths(
            vertices=vertices,
            objects=objects,
            mask=self.masks[..., order],
            confidence_threshold=self.confidence_threshold,
            interaction_types=self.interaction_types[..., :order],
        )

    def reshape(self, *batch: int) -> Self:
        """
        Return a new paths instance with reshaped paths' batch dimensions to match a given shape.

        Args:
            batch: New batch shape.

        Returns:
            A new paths instance with specified batch dimensions.
        """
        vertices = self.vertices.reshape(*batch, self.path_length, 3)
        resolved_batch = vertices.shape[:-2]
        objects = self.objects.reshape(*resolved_batch, self.path_length)
        masks = self.masks.reshape(*resolved_batch, self.masks.shape[-1])
        interaction_types = self.interaction_types.reshape(
            *resolved_batch, self.path_length - 2
        )

        return eqx.tree_at(
            lambda p: (
                p.vertices,
                p.objects,
                p.masks,
                p.interaction_types,
            ),
            self,
            (vertices, objects, masks, interaction_types),
        )

    def squeeze(self, axis: int | Sequence[int] | None = None) -> Self:
        """
        Return a new paths instance by squeezing one or more axes of paths' batch dimensions.

        Args:
            axis: See :func:`jax.numpy.squeeze` for allowed values.

        Returns:
            A new paths instance with squeezed batch dimensions.

        Raises:
            ValueError: If one of the provided axes is out-of-bounds,
                or if trying to squeeze a 0-dimensional batch.
        """
        ndim = self.vertices.ndim - 2
        if axis is not None and ndim == 0:
            msg = "Cannot squeeze a 0-dimensional batch!"
            raise ValueError(msg)
        if isinstance(axis, int):
            axis = (axis,)
        if isinstance(axis, Sequence):
            axis = tuple(a + ndim if a < 0 else a for a in axis)

            if any(ax >= ndim or ax < 0 for ax in axis):
                msg = "One of the provided axes is out-of-bounds!"
                raise ValueError(msg)

        vertices = self.vertices.squeeze(axis)
        objects = self.objects.squeeze(axis)
        masks = self.masks.squeeze(axis)
        interaction_types = self.interaction_types.squeeze(axis)

        return eqx.tree_at(
            lambda p: (
                p.vertices,
                p.objects,
                p.masks,
                p.interaction_types,
            ),
            self,
            (vertices, objects, masks, interaction_types),
        )

    def __iter__(self) -> Iterator["TracedPaths"]:
        """
        Return an iterator over the highest-order masked paths.

        Yields:
            The highest-order masked paths.
        """
        yield from self.get_paths(self.order)

    def masked(self) -> TracedPaths:
        """
        Return a flattened :class:`TracedPaths` instance keeping only valid highest-order paths.

        Returns:
            A flattened trace paths instance with only valid paths.
        """
        return self.get_paths(self.order).masked()

    @property
    def masked_vertices(self) -> Float[Array, "num_valid_paths path_length 3"]:
        """The array of masked vertices of the highest-order paths, with batched dimensions flattened into one."""
        return self.get_paths(self.order).masked_vertices

    @property
    def masked_objects(self) -> Int[Array, "num_valid_paths path_length"]:
        """The array of masked objects of the highest-order paths, with batched dimensions flattened into one."""
        return self.get_paths(self.order).masked_objects

    def plot(self, **kwargs: Any) -> PlotOutput:
        """
        Plot the paths on a 3D scene.

        Args:
            kwargs: Keyword arguments passed to plotting backend.

        Returns:
            The resulting plot output.
        """
        with reuse(**kwargs, pass_all_kwargs=True) as output:
            for order in range(self.order + 1):
                self.get_paths(order).plot()

        return output


# Deprecated alias
class SBRPaths(LaunchedPaths):
    """
    Deprecated alias for :class:`LaunchedPaths`.

    .. deprecated:: 0.10
        Use :class:`LaunchedPaths` instead.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "SBRPaths is deprecated, use LaunchedPaths instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
