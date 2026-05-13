import math
import typing
from collections.abc import Callable, Iterator, Sized
from functools import cache, partial
from typing import TYPE_CHECKING, Any, TypeVar, overload

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from differt.geometry import fibonacci_lattice, viewing_frustum
from differt.utils import smoothing_function
from differt_core.rt import CompleteGraph

if TYPE_CHECKING or hasattr(typing, "GENERATING_DOCS"):
    from typing import Self
else:
    Self = Any  # Because runtime type checking from 'beartype' will fail when combined with 'jaxtyping'

_T = TypeVar("_T")


class SizedIterator(Iterator[_T], Sized):
    """A custom generic class that is both :class:`Iterator<collections.abc.Iterator>` and :class:`Sized<collections.abc.Sized>`.

    The main purpose of this class is to be able to use
    `tqdm <https://github.com/tqdm/tqdm>`_ utilities
    on iterators and have some meaningful information about how iterations are left.

    Args:
        iter_: The iterator.
        size: The size, i.e., length, of the iterator, or a callable that returns its current length.

    Examples:
        The following example shows how to create a sized iterator.

        >>> from differt.rt import SizedIterator
        >>> l = [1, 2, 3, 4, 5]
        >>> it = SizedIterator(iter=iter(l), size=5)
        >>> len(it)
        5
        >>> it = SizedIterator(iter=iter(l), size=l.__len__)
        >>> len(it)
        5

    """

    __slots__ = ("_iter", "_size")

    def __init__(self, iter: Iterator[_T], size: int | Callable[[], int]) -> None:  # noqa: A002
        self._iter = iter
        self._size = size

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> _T:
        return next(self._iter)

    def __len__(self) -> int:
        if isinstance(self._size, int):
            return self._size
        return self._size()


@cache
def generate_all_path_candidates(
    num_primitives: int,
    order: int,
) -> Int[Array, "num_candidates order"]:
    """
    Generate an array of all path candidates for fixed path order and a number of primitives.

    The returned array contains, for each row, an array of
    ``order`` indices indicating the primitive with which the path interacts.

    This list is generated as the list of all paths from one node to
    another, by passing by exactly ``order`` primitives. Calling this function
    is equivalent to calling :func:`itertools.product` with parameters
    ``[0, 1, ..., num_primitives - 1]`` and ``repeat=order``, and removing entries
    containing loops, i.e., two or more consecutive indices that are equal.

    Args:
        num_primitives: The (positive) number of primitives.
        order: The path order. An order less than one returns an empty array.

    Returns:
        An unsigned array with primitive indices on each column. Its number of
        columns is actually equal to
        ``num_primitives * ((num_primitives - 1) ** (order - 1))``.
    """
    return jnp.asarray(
        CompleteGraph(num_primitives).all_paths_array(
            from_=num_primitives,
            to=num_primitives + 1,
            depth=order + 2,
            include_from_and_to=False,
        ),
        dtype=int,
    )


def generate_all_path_candidates_iter(
    num_primitives: int,
    order: int,
) -> SizedIterator[Int[Array, " order"]]:
    """
    Iterator variant of :func:`generate_all_path_candidates`.

    Args:
        num_primitives: The (positive) number of primitives.
        order: The path order.

    Returns:
        An iterator of unsigned arrays with primitive indices.
    """
    it = CompleteGraph(num_primitives).all_paths(
        from_=num_primitives,
        to=num_primitives + 1,
        depth=order + 2,
        include_from_and_to=False,
    )
    m = (jnp.asarray(arr, dtype=int) for arr in it)
    return SizedIterator(m, size=it.__len__)


def generate_all_path_candidates_chunks_iter(
    num_primitives: int,
    order: int,
    chunk_size: int = 1000,
) -> SizedIterator[Int[Array, "chunk_size order"]]:
    """
    Iterator variant of :func:`generate_all_path_candidates`, grouped in chunks of size of max. ``chunk_size``.

    Args:
        num_primitives: The (positive) number of primitives.
        order: The path order.
        chunk_size: The size of each chunk.

    Returns:
        An iterator of unsigned arrays with primitive indices.
    """
    it = CompleteGraph(num_primitives).all_paths_array_chunks(
        from_=num_primitives,
        to=num_primitives + 1,
        depth=order + 2,
        include_from_and_to=False,
        chunk_size=chunk_size,
    )
    m = (jnp.asarray(arr, dtype=int) for arr in it)
    return SizedIterator(m, size=it.__len__)


@overload
def rays_intersect_triangles(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch 3 3"],
    *,
    epsilon: Float[ArrayLike, ""] | None = ...,
    smoothing_factor: Float[ArrayLike, ""],
) -> tuple[Float[Array, " *batch"], Bool[Array, " *batch"]]: ...


@overload
def rays_intersect_triangles(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch 3 3"],
    *,
    epsilon: Float[ArrayLike, ""] | None = ...,
    smoothing_factor: None = ...,
) -> tuple[Float[Array, " *batch"], Float[Array, " *batch"]]: ...


@eqx.filter_jit
def rays_intersect_triangles(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch 3 3"],
    *,
    epsilon: Float[ArrayLike, ""] | None = None,
    smoothing_factor: Float[ArrayLike, ""] | None = None,
) -> tuple[Float[Array, " *batch"], Bool[Array, " *batch"] | Float[Array, " *batch"]]:
    """
    Return whether rays intersect corresponding triangles using the Möller-Trumbore algorithm.

    The current implementation closely follows the C++ code from Wikipedia.

    Args:
        ray_origins: An array of origin vertices.
        ray_directions: An array of ray directions. The ray ends
            should be equal to ``ray_origins + ray_directions``.
        triangle_vertices: An array of triangle vertices.
        epsilon: A small tolerance threshold that allows rays
            to hit the triangles slightly outside the actual area.
            A positive value virtually increases the size of the triangles,
            a negative value will have the opposite effect.

            Such a tolerance is especially useful when rays are hitting
            triangle edges, a very common case if geometries are planes
            split into multiple triangles.

            If not specified, the default is ten times the epsilon value
            of the currently used floating point dtype.
        smoothing_factor: If set, hard conditions are replaced with smoothed ones,
            as described in :cite:`fully-eucap2024`, and this argument parameterizes the slope
            of the smoothing function. The second output value is now a real value
            between 0 (:data:`False`) and 1 (:data:`True`).

            For more details, refer to :ref:`smoothing`.

    Returns:
        For each ray, return the scale factor of ``ray_directions`` for the
        vector to reach the corresponding triangle, and whether the intersection
        actually lies inside the triangle.

    Examples:
        The following example shows how to identify triangles that are
        intersected by rays.

        .. plotly::

            >>> import equinox as eqx
            >>> from differt.geometry import fibonacci_lattice
            >>> from differt.plotting import draw_rays
            >>> from differt.rt import (
            ...     rays_intersect_triangles,
            ... )
            >>> from differt.scene import (
            ...     get_sionna_scene,
            ...     download_sionna_scenes,
            ... )
            >>> from differt.scene import TriangleScene
            >>>
            >>> download_sionna_scenes()
            >>> file = get_sionna_scene("simple_street_canyon")
            >>> scene = TriangleScene.load_xml(file)
            >>> scene = eqx.tree_at(
            ...     lambda s: s.transmitters, scene, jnp.array([-33, 0, 32.0])
            ... )
            >>> ray_origins, ray_directions = jnp.broadcast_arrays(
            ...     scene.transmitters, fibonacci_lattice(25)
            ... )
            >>> # [num_rays=25 num_triangles]
            >>> t, hit = rays_intersect_triangles(
            ...     ray_origins[:, None, :],
            ...     ray_directions[:, None, :],
            ...     scene.mesh.triangle_vertices,
            ... )
            >>> rays_hit = hit.any(axis=1)  # True if rays hit any triangle
            >>> triangles_hit = hit.any(axis=0)  # True if triangles hit by any ray
            >>> ray_directions *= jnp.max(
            ...     t, axis=1, keepdims=True, initial=1.0, where=hit
            ... )  # Scale rays length before plotting
            >>> fig = draw_rays(  # We only plot rays hitting at least one triangle
            ...     ray_origins[rays_hit, :],
            ...     ray_directions[rays_hit, :],
            ...     backend="plotly",
            ...     color="red",
            ...     showlegend=False,
            ... )
            >>> visible_color = jnp.array([0.2, 0.2, 0.2])
            >>> scene = eqx.tree_at(
            ...     lambda s: s.mesh.face_colors,
            ...     scene,
            ...     scene.mesh.face_colors.at[triangles_hit, :].set(visible_color),
            ... )
            >>> fig = scene.plot(backend="plotly", figure=fig, showlegend=False)
            >>> fig  # doctest: +SKIP
    """
    ray_origins = jnp.asarray(ray_origins)
    ray_directions = jnp.asarray(ray_directions)
    triangle_vertices = jnp.asarray(triangle_vertices)

    if epsilon is None:
        dtype = jnp.result_type(ray_origins, ray_directions, triangle_vertices)
        epsilon = 10 * jnp.finfo(dtype).eps

    epsilon = jnp.asarray(epsilon)

    # [*batch 3]
    vertex_0 = triangle_vertices[..., 0, :]
    vertex_1 = triangle_vertices[..., 1, :]
    vertex_2 = triangle_vertices[..., 2, :]

    # [*batch 3]
    edge_1 = vertex_1 - vertex_0
    edge_2 = vertex_2 - vertex_0

    # [*batch 3]
    h = jnp.cross(ray_directions, edge_2)

    # [*batch]
    a = jnp.sum(h * edge_1, axis=-1)
    a: Array = jnp.where(a == 0.0, jnp.inf, a)  # Avoid division by zero

    if smoothing_factor is not None:
        hit = smoothing_function(jnp.abs(a) - epsilon, smoothing_factor)
    else:
        hit = jnp.abs(a) > epsilon

    f = 1.0 / a
    s = ray_origins - vertex_0
    u = f * jnp.sum(s * h, axis=-1)

    if smoothing_factor is not None:
        hit = jnp.stack(
            (
                hit,
                smoothing_function(u - 0.0, smoothing_factor),
                smoothing_function(1.0 - u, smoothing_factor),
            ),
            axis=-1,
        ).min(axis=-1, initial=1.0)
    else:
        hit &= (u >= 0.0) & (u <= 1.0)

    q = jnp.cross(s, edge_1)
    v = f * jnp.sum(q * ray_directions, axis=-1)

    if smoothing_factor is not None:
        hit = jnp.stack(
            (
                hit,
                smoothing_function(v - 0.0, smoothing_factor),
                smoothing_function(1.0 - (u + v), smoothing_factor),
            ),
            axis=-1,
        ).min(axis=-1, initial=1.0)
    else:
        hit &= (v >= 0.0) & (u + v <= 1.0)

    t = f * jnp.sum(q * edge_2, axis=-1)

    if smoothing_factor is not None:
        hit = jnp.minimum(hit, smoothing_function(t - epsilon, smoothing_factor))
    else:
        hit &= t > epsilon

    return t, hit


def _ray_intersect_any_triangle_batched(
    ray_origin: Float[Array, "3"],
    ray_direction: Float[Array, "3"],
    triangle_vertices: Float[Array, "num_triangles 3 3"],
    active_triangles: Bool[Array, " num_triangles"] | None,
    *,
    hit_threshold: Float[Array, ""] | None,
    smoothing_factor: Float[ArrayLike, ""] | None,
    **kwargs: Any,
) -> Bool[Array, ""] | Float[Array, ""]:
    ts, hits = jax.vmap(
        partial(rays_intersect_triangles, smoothing_factor=smoothing_factor, **kwargs),
        in_axes=(None, None, 0),
    )(ray_origin, ray_direction, triangle_vertices)
    if smoothing_factor is not None:
        return jnp.minimum(
            hits, smoothing_function(hit_threshold - ts, smoothing_factor)
        ).max(axis=-1, where=active_triangles)
    return ((ts < hit_threshold) & hits).any(axis=-1, where=active_triangles)


def _ray_intersect_any_triangle(
    ray_origin: Float[Array, "3"],
    ray_direction: Float[Array, "3"],
    triangle_vertices: Float[Array, "num_triangles 3 3"],
    active_triangles: Bool[Array, " num_triangles"] | None,
    *,
    hit_threshold: Float[Array, ""] | None,
    smoothing_factor: Float[ArrayLike, ""] | None,
    batch_size: int,
    **kwargs: Any,
) -> Bool[Array, ""] | Float[Array, ""]:
    def reduce_fn(
        left: Bool[Array, ""] | Float[Array, ""],
        right: Bool[Array, ""] | Float[Array, ""],
    ) -> Bool[Array, ""] | Float[Array, ""]:
        if smoothing_factor is not None:
            return jnp.maximum(left, right)
        return left | right

    def body_fn(
        batch_index: Int[Array, ""],
        intersect_so_far: Bool[Array, ""] | Float[Array, ""],
    ) -> Bool[Array, ""] | Float[Array, ""]:
        start_index = batch_index * batch_size
        batch_of_triangle_vertices = jax.lax.dynamic_slice_in_dim(
            triangle_vertices, start_index, batch_size, axis=0
        )
        batch_of_active_triangles = (
            jax.lax.dynamic_slice_in_dim(
                active_triangles, start_index, batch_size, axis=0
            )
            if active_triangles is not None
            else None
        )
        intersect_in_batch = _ray_intersect_any_triangle_batched(
            ray_origin,
            ray_direction,
            batch_of_triangle_vertices,
            batch_of_active_triangles,
            hit_threshold=hit_threshold,
            smoothing_factor=smoothing_factor,
            **kwargs,
        )

        return reduce_fn(intersect_so_far, intersect_in_batch)

    num_triangles = triangle_vertices.shape[0]
    num_batches, rem = divmod(num_triangles, batch_size)

    init_val = 0.0 if smoothing_factor is not None else False

    intersect = jax.lax.fori_loop(
        0,
        num_batches,
        body_fn,
        init_val=init_val,
    )

    if rem > 0:
        intersect = reduce_fn(
            intersect,
            _ray_intersect_any_triangle_batched(
                ray_origin,
                ray_direction,
                triangle_vertices[-rem:, :],
                (active_triangles[-rem:] if active_triangles is not None else None),
                hit_threshold=hit_threshold,
                smoothing_factor=smoothing_factor,
                **kwargs,
            ),
        )

    return intersect


def _first_triangle_hit_by_ray_batched(
    ray_origin,
    ray_direction,
    triangle_vertices,
    active_triangles,
    triangle_indices,
    *,
    dist_tol,
    **kwargs,
):
    ts, hits = jax.vmap(
        partial(rays_intersect_triangles, **kwargs),
        in_axes=(None, None, 0),
    )(ray_origin, ray_direction, triangle_vertices)

    if active_triangles is not None:
        hits &= active_triangles

    ts = jnp.where(hits, ts, jnp.inf)
    min_t = jnp.min(ts)

    center_distances = jnp.linalg.norm(
        triangle_vertices.mean(axis=-2) - ray_origin, axis=-1
    )

    is_close = jnp.abs(ts - min_t) < dist_tol
    dist_to_check = jnp.where(is_close, center_distances, jnp.inf)
    min_dist = jnp.min(dist_to_check)

    is_best = (dist_to_check == min_dist) & is_close
    best_idx = jnp.argmax(is_best)

    any_hit = jnp.any(hits)

    return (
        jnp.where(any_hit, triangle_indices[best_idx], -1),
        jnp.where(any_hit, min_t, jnp.inf),
        jnp.where(any_hit, min_dist, jnp.inf),
    )


def _first_triangle_hit_by_ray(
    ray_origin,
    ray_direction,
    triangle_vertices,
    active_triangles,
    *,
    batch_size,
    dist_tol,
    **kwargs,
):
    def combine_best(
        left: tuple[Int[Array, ""], Float[Array, ""], Float[Array, ""]],
        right: tuple[Int[Array, ""], Float[Array, ""], Float[Array, ""]],
    ) -> tuple[Int[Array, ""], Float[Array, ""], Float[Array, ""]]:
        idx1, t1, d1 = left
        idx2, t2, d2 = right

        cond = jnp.where(
            jnp.abs(t1 - t2) < dist_tol,
            d1 < d2,
            t1 < t2,
        )

        return (
            jnp.where(cond, idx1, idx2),
            jnp.where(cond, t1, t2),
            jnp.where(cond, d1, d2),
        )

    def body_fn(
        batch_index: Int[Array, ""],
        best_so_far: tuple[Int[Array, ""], Float[Array, ""], Float[Array, ""]],
    ) -> tuple[Int[Array, ""], Float[Array, ""], Float[Array, ""]]:
        start_index = batch_index * batch_size
        batch_of_triangle_vertices = jax.lax.dynamic_slice_in_dim(
            triangle_vertices, start_index, batch_size, axis=0
        )
        batch_of_active_triangles = (
            jax.lax.dynamic_slice_in_dim(
                active_triangles, start_index, batch_size, axis=0
            )
            if active_triangles is not None
            else None
        )
        batch_of_indices = jnp.arange(batch_size, dtype=jnp.int32) + start_index

        best_in_batch = _first_triangle_hit_by_ray_batched(
            ray_origin,
            ray_direction,
            batch_of_triangle_vertices,
            batch_of_active_triangles,
            batch_of_indices,
            dist_tol=dist_tol,
            **kwargs,
        )

        return combine_best(best_so_far, best_in_batch)

    num_triangles = triangle_vertices.shape[0]
    num_batches, rem = divmod(num_triangles, batch_size)

    init_val = (
        jnp.array(-1, dtype=jnp.int32),
        jnp.array(jnp.inf, dtype=ray_origin.dtype),
        jnp.array(jnp.inf, dtype=ray_origin.dtype),
    )

    best = jax.lax.fori_loop(
        0,
        num_batches,
        body_fn,
        init_val=init_val,
    )

    if rem > 0:
        start_index = num_batches * batch_size
        best = combine_best(
            best,
            _first_triangle_hit_by_ray_batched(
                ray_origin,
                ray_direction,
                triangle_vertices[-rem:, :],
                (active_triangles[-rem:] if active_triangles is not None else None),
                jnp.arange(rem, dtype=jnp.int32) + start_index,
                dist_tol=dist_tol,
                **kwargs,
            ),
        )

    return best[0], best[1]


@overload
def rays_intersect_any_triangle(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch num_triangles 3 3"],
    active_triangles: Bool[ArrayLike, "*#batch num_triangles"] | None = ...,
    *,
    hit_tol: Float[ArrayLike, ""] | None = ...,
    smoothing_factor: None = ...,
    batch_size: int | None = ...,
    ray_batch_size: int | None = ...,
    tri_batch_size: int | None = ...,
    **kwargs: Any,
) -> Bool[Array, " *batch"]: ...


@overload
def rays_intersect_any_triangle(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch num_triangles 3 3"],
    active_triangles: Bool[ArrayLike, "*#batch num_triangles"] | None = ...,
    *,
    hit_tol: Float[ArrayLike, ""] | None = ...,
    smoothing_factor: Float[ArrayLike, ""],
    batch_size: int | None = ...,
    ray_batch_size: int | None = ...,
    tri_batch_size: int | None = ...,
    **kwargs: Any,
) -> Float[Array, " *batch"]: ...


@eqx.filter_jit
def rays_intersect_any_triangle(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch num_triangles 3 3"],
    active_triangles: Bool[ArrayLike, "*#batch num_triangles"] | None = None,
    *,
    hit_tol: Float[ArrayLike, ""] | None = None,
    smoothing_factor: Float[ArrayLike, ""] | None = None,
    batch_size: int | None = 4096,
    ray_batch_size: int | None = None,
    tri_batch_size: int | None = None,
    **kwargs: Any,
) -> Bool[Array, " *batch"] | Float[Array, " *batch"]:
    """
    Return whether rays intersect any of the triangles using the Möller-Trumbore algorithm.

    This function should be used when allocating an array of size
    ``*batch num_triangles 3`` (or bigger) is not possible, and you are only interested in
    checking if at least one of the triangles is intersected.

    A triangle is considered to be intersected if
    ``t < (1 - hit_tol) & hit`` evaluates to :data:`True`.

    Args:
        ray_origins: An array of origin vertices.
        ray_directions: An array of ray direction. The ray ends
            should be equal to ``ray_origins + ray_directions``.
        triangle_vertices: An array of triangle vertices.
        active_triangles: An optional array of boolean values indicating
            which triangles are active, i.e., should be considered for intersection.

            If not specified, all triangles are considered active.
        hit_tol: The tolerance applied to check if a ray hits another object or not,
            before it reaches the expected position, i.e., the 'interaction' object.

            Using a non-zero tolerance is required as it would otherwise trigger
            false positives.

            If not specified, the default is ten times the epsilon value
            of the currently used floating point dtype.
        smoothing_factor: If set, hard conditions are replaced with smoothed ones,
            as described in :cite:`fully-eucap2024`, and this argument parameterizes the slope
            of the smoothing function. The second output value is now a real value
            between 0 (:data:`False`) and 1 (:data:`True`).

            For more details, refer to :ref:`smoothing`.
        batch_size: The default batch size used when either ``ray_batch_size`` or
            ``tri_batch_size`` is not specified. This allows to make a trade-off between memory
            usage and performance.

            If :data:`None`, the provided ``ray_batch_size`` and ``tri_batch_size`` values are
            used. Otherwise, this value is used as the default for whichever of
            ``ray_batch_size`` and ``tri_batch_size`` are left unspecified.
        ray_batch_size: The number of rays to process in a single batch.
            This allows to make a trade-off between memory usage and performance.

            The ray batch size is automatically adjusted to be the minimum of the number of rays
            and the specified ray batch size.

            If :data:`None`, it defaults to ``batch_size``.
        tri_batch_size: The number of triangles to process in a single batch.
            This allows to make a trade-off between memory usage and performance.

            The triangle batch size is automatically adjusted to be the minimum of the number of
            triangles and the specified triangle batch size.

            If :data:`None`, it defaults to ``batch_size``.
        kwargs: Keyword arguments passed to
            :func:`rays_intersect_triangles`.

    Returns:
        For each ray, whether it intersects with any of the triangles.
    """
    ray_origins = jnp.asarray(ray_origins)
    ray_directions = jnp.asarray(ray_directions)
    triangle_vertices = jnp.asarray(triangle_vertices)

    if hit_tol is None:
        dtype = jnp.result_type(ray_origins, ray_directions, triangle_vertices)
        hit_tol = 10.0 * jnp.finfo(dtype).eps

    hit_threshold = 1.0 - jnp.asarray(hit_tol)

    num_triangles = triangle_vertices.shape[-3]

    if active_triangles is not None:
        active_triangles = jnp.asarray(active_triangles)

    batch = jnp.broadcast_shapes(
        ray_origins.shape[:-1],
        ray_directions.shape[:-1],
        triangle_vertices.shape[:-3],
        active_triangles.shape[:-1] if active_triangles is not None else (),
    )
    num_rays = math.prod(batch)
    ray_batch_size = (
        ray_batch_size
        if ray_batch_size is not None
        else batch_size
        if batch_size is not None
        else num_rays
    )
    ray_batch_size = min(ray_batch_size, num_rays)
    tri_batch_size = (
        tri_batch_size
        if tri_batch_size is not None
        else batch_size
        if batch_size is not None
        else num_triangles
    )
    tri_batch_size = min(tri_batch_size, num_triangles)
    if num_triangles == 0:
        # If there are no triangles, there are no intersections
        return (
            jnp.zeros(
                batch,
                dtype=jnp.result_type(ray_origins, ray_directions, triangle_vertices),
            )
            if smoothing_factor is not None
            else jnp.zeros(batch, dtype=bool)
        )

    if num_rays > ray_batch_size:
        xs = []
        argnames = []
        map_fn = partial(
            _ray_intersect_any_triangle,
            hit_threshold=hit_threshold,
            smoothing_factor=smoothing_factor,
            batch_size=tri_batch_size,
            **kwargs,
        )
        if math.prod(ray_origins.shape[:-1]) > 1:
            ray_origins = jnp.broadcast_to(ray_origins, (*batch, 3))
            xs.append(ray_origins.reshape(-1, 3))
            argnames.append("ray_origin")
        else:
            map_fn = partial(map_fn, ray_origin=ray_origins)
        if math.prod(ray_directions.shape[:-1]) > 1:
            ray_directions = jnp.broadcast_to(ray_directions, (*batch, 3))
            xs.append(ray_directions.reshape(-1, 3))
            argnames.append("ray_direction")
        else:
            map_fn = partial(map_fn, ray_direction=ray_directions)
        if math.prod(triangle_vertices.shape[:-3]) > 1:
            triangle_vertices = jnp.broadcast_to(
                triangle_vertices, (*batch, num_triangles, 3, 3)
            )
            xs.append(triangle_vertices.reshape(-1, num_triangles, 3, 3))
            argnames.append("triangle_vertices")
        else:
            map_fn = partial(map_fn, triangle_vertices=triangle_vertices)
        if active_triangles is not None and math.prod(active_triangles.shape[:-1]) > 1:
            active_triangles = jnp.broadcast_to(
                active_triangles, (*batch, num_triangles)
            )
            xs.append(active_triangles.reshape(-1, num_triangles))
            argnames.append("active_triangles")
        else:
            map_fn = partial(map_fn, active_triangles=active_triangles)

        def f(args):
            return map_fn(**dict(zip(argnames, args, strict=True)))

        return jax.lax.map(f, tuple(xs), batch_size=ray_batch_size).reshape(batch)
    return jnp.vectorize(
        partial(
            _ray_intersect_any_triangle,
            hit_threshold=hit_threshold,
            smoothing_factor=smoothing_factor,
            batch_size=tri_batch_size,
            **kwargs,
        ),
        signature="(3),(3),(n,3,3),(n)->()"
        if active_triangles is not None
        else "(3),(3),(n,3,3),()->()",
    )(ray_origins, ray_directions, triangle_vertices, active_triangles)


def _triangles_visible_from_vertex(
    vertex,
    triangle_vertices,
    active_triangles,
    *,
    num_rays,
    num_triangles,
    ray_batch_size,
    tri_batch_size,
    world_vertices,
    active_vertices,
    **kwargs,
):
    frustum = viewing_frustum(vertex, world_vertices, active_vertices=active_vertices)
    ray_directions = fibonacci_lattice(num_rays, frustum=frustum)

    indices, _ = first_triangles_hit_by_rays(
        vertex,
        ray_directions,
        triangle_vertices,
        active_triangles=active_triangles,
        ray_batch_size=ray_batch_size,
        tri_batch_size=tri_batch_size,
        **kwargs,
    )

    valid_hits = indices >= 0
    safe_indices = jnp.where(valid_hits, indices, 0)
    visible = jnp.zeros(num_triangles, dtype=bool).at[safe_indices].max(valid_hits)

    return visible


@eqx.filter_jit
def triangles_visible_from_vertices(
    vertices: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch num_triangles 3 3"],
    active_triangles: Bool[ArrayLike, "*#batch num_triangles"] | None = None,
    num_rays: int = int(1e6),
    batch_size: int | None = 4096,
    ray_batch_size: int | None = None,
    tri_batch_size: int | None = None,
    **kwargs: Any,
) -> Bool[Array, "*batch num_triangles"]:
    """
    Return whether triangles are visible from vertex positions.

    This function uses ray launching and
    :func:`fibonacci_lattice<differt.geometry.fibonacci_lattice>` to estimate
    whether a given triangle can be reached from a specific vertex, i.e., with a ray path,
    without interacting with any other triangle facet.

    It also uses
    :func:`viewing_frustum<differt.geometry.viewing_frustum>` to only
    launch rays in a spatial region that contains triangles.

    Args:
        vertices: An array of vertices, used as origins of the rays.

            Usually, this would be an array of transmitter positions.
        triangle_vertices: An array of triangle vertices.
        active_triangles: An optional array of boolean values indicating
            which triangles are active, i.e., should be considered for intersection.

            If not specified, all triangles are considered active.
        num_rays: The number of rays to launch.

            The larger, the more accurate.
        batch_size: The default batch size used when either ``ray_batch_size`` or
            ``tri_batch_size`` is not specified. This allows to make a trade-off between memory
            usage and performance.

            If :data:`None`, the provided ``ray_batch_size`` and ``tri_batch_size`` values are
            used. Otherwise, this value is used as the default for whichever of
            ``ray_batch_size`` and ``tri_batch_size`` are left unspecified.
        ray_batch_size: The number of rays to process in a single batch.
            This allows to make a trade-off between memory usage and performance.

            The ray batch size is automatically adjusted to be the minimum of the number of rays
            and the specified ray batch size.

            If :data:`None`, it defaults to ``batch_size``.
        tri_batch_size: The number of triangles to process in a single batch.
            This allows to make a trade-off between memory usage and performance.

            The triangle batch size is automatically adjusted to be the minimum of the number of
            triangles and the specified triangle batch size.

            If :data:`None`, it defaults to ``batch_size``.
        kwargs: Keyword arguments passed to
            :func:`rays_intersect_triangles`.

    Returns:
        For each triangle, whether it intersects with any of the rays.

    Examples:
        The following example shows how to identify triangles as
        visible from a given transmitter, coloring them in dark gray.

        .. plotly::
            :context: reset

            >>> import equinox as eqx
            >>> from differt.rt import (
            ...     triangles_visible_from_vertices,
            ... )
            >>> from differt.scene import (
            ...     TriangleScene,
            ...     get_sionna_scene,
            ...     download_sionna_scenes,
            ... )
            >>>
            >>> download_sionna_scenes()
            >>> file = get_sionna_scene("simple_street_canyon")
            >>> scene = TriangleScene.load_xml(file)
            >>> scene = eqx.tree_at(
            ...     lambda s: s.transmitters, scene, jnp.array([-33, 0, 32.0])
            ... )
            >>> visible_triangles = triangles_visible_from_vertices(
            ...     scene.transmitters,
            ...     scene.mesh.triangle_vertices,
            ... )
            >>> visible_color = jnp.array([0.2, 0.2, 0.2])
            >>> scene = eqx.tree_at(
            ...     lambda s: s.mesh.face_colors,
            ...     scene,
            ...     scene.mesh.face_colors.at[visible_triangles, :].set(visible_color),
            ... )
            >>> fig = scene.plot(backend="plotly")
            >>> fig  # doctest: +SKIP

        In this example, a receiver is placed at the opposite side of the street canyon,
        and its visible triangles are colored in blue. Triangles that are visible from both
        the transmitter and the receiver are colored in yellow.

        .. plotly::
            :context:

            >>> scene = eqx.tree_at(
            ...     lambda s: s.receivers, scene, jnp.array([33, 0, 1.5])
            ... )
            >>> visible_triangles = triangles_visible_from_vertices(
            ...     jnp.stack((scene.transmitters, scene.receivers)),
            ...     scene.mesh.triangle_vertices,
            ... )
            >>> triangles_visible_from_tx = visible_triangles[0, :]
            >>> triangles_visible_from_rx = visible_triangles[1, :]
            >>> visible_by_tx_color = jnp.array([0.2, 0.2, 0.2])
            >>> visible_by_rx_color = jnp.array([0.2, 0.8, 0.2])
            >>> visible_by_both_color = jnp.array([0.8, 0.8, 0.2])
            >>> scene = eqx.tree_at(
            ...     lambda s: s.mesh.face_colors,
            ...     scene,
            ...     scene.mesh.face_colors.at[triangles_visible_from_tx, :].set(
            ...         visible_by_tx_color
            ...     ),
            ... )
            >>> scene = eqx.tree_at(
            ...     lambda s: s.mesh.face_colors,
            ...     scene,
            ...     scene.mesh.face_colors.at[triangles_visible_from_rx, :].set(
            ...         visible_by_rx_color
            ...     ),
            ... )
            >>> scene = eqx.tree_at(
            ...     lambda s: s.mesh.face_colors,
            ...     scene,
            ...     scene.mesh.face_colors.at[
            ...         triangles_visible_from_tx & triangles_visible_from_rx, :
            ...     ].set(visible_by_both_color),
            ... )
            >>> fig = scene.plot(backend="plotly")
            >>> fig  # doctest: +SKIP
    """
    vertices = jnp.asarray(vertices)
    triangle_vertices = jnp.asarray(triangle_vertices)

    num_triangles = triangle_vertices.shape[-3]

    if active_triangles is not None:
        active_triangles = jnp.asarray(active_triangles)
        active_vertices = jnp.repeat(active_triangles, 4, axis=-1)
    else:
        active_vertices = None

    triangle_centers = triangle_vertices.mean(axis=-2, keepdims=True)
    world_vertices = jnp.concat((triangle_vertices, triangle_centers), axis=-2).reshape(
        *triangle_vertices.shape[:-3], -1, 3
    )

    batch = jnp.broadcast_shapes(
        vertices.shape[:-1],
        triangle_vertices.shape[:-3],
        active_triangles.shape[:-1] if active_triangles is not None else (),
    )
    num_vertices = math.prod(batch)

    ray_batch_size = (
        ray_batch_size
        if ray_batch_size is not None
        else batch_size
        if batch_size is not None
        else num_rays
    )
    ray_batch_size = min(ray_batch_size, num_rays)
    tri_batch_size = (
        tri_batch_size
        if tri_batch_size is not None
        else batch_size
        if batch_size is not None
        else num_triangles
    )
    tri_batch_size = min(tri_batch_size, num_triangles)

    if num_triangles == 0:
        return jnp.zeros((*batch, 0), dtype=jnp.bool_)

    xs = []
    argnames = []
    map_fn = partial(
        _triangles_visible_from_vertex,
        num_rays=num_rays,
        num_triangles=num_triangles,
        ray_batch_size=ray_batch_size,
        tri_batch_size=tri_batch_size,
        **kwargs,
    )
    if math.prod(vertices.shape[:-1]) > 1:
        xs.append(vertices.reshape(-1, 3))
        argnames.append("vertex")
    else:
        map_fn = partial(map_fn, vertex=vertices.reshape(3))
    if math.prod(triangle_vertices.shape[:-3]) > 1:
        xs.append(triangle_vertices.reshape(-1, num_triangles, 3, 3))
        argnames.append("triangle_vertices")
    else:
        map_fn = partial(
            map_fn, triangle_vertices=triangle_vertices.reshape(num_triangles, 3, 3)
        )
    if active_triangles is not None and math.prod(active_triangles.shape[:-1]) > 1:
        xs.append(active_triangles.reshape(-1, num_triangles))
        argnames.append("active_triangles")
    else:
        map_fn = partial(
            map_fn,
            active_triangles=active_triangles.reshape(num_triangles)
            if active_triangles is not None
            else None,
        )

    # Also need to handle world_vertices and active_vertices broadcasting
    if math.prod(world_vertices.shape[:-2]) > 1:
        xs.append(world_vertices.reshape(-1, *world_vertices.shape[-2:]))
        argnames.append("world_vertices")
    else:
        map_fn = partial(map_fn, world_vertices=world_vertices.reshape(-1, 3))

    if active_vertices is not None and math.prod(active_vertices.shape[:-1]) > 1:
        xs.append(active_vertices.reshape(-1, active_vertices.shape[-1]))
        argnames.append("active_vertices")
    else:
        map_fn = partial(
            map_fn,
            active_vertices=active_vertices.reshape(-1)
            if active_vertices is not None
            else None,
        )

    def f(args):
        return map_fn(**dict(zip(argnames, args, strict=True)))

    if not xs:
        # Case where everything is shared or num_vertices == 1
        return _triangles_visible_from_vertex(
            vertices.reshape(3),
            triangle_vertices.reshape(num_triangles, 3, 3),
            active_triangles.reshape(num_triangles)
            if active_triangles is not None
            else None,
            num_rays=num_rays,
            num_triangles=num_triangles,
            ray_batch_size=ray_batch_size,
            tri_batch_size=tri_batch_size,
            world_vertices=world_vertices.reshape(-1, 3),
            active_vertices=active_vertices.reshape(-1)
            if active_vertices is not None
            else None,
            **kwargs,
        ).reshape((*batch, num_triangles))

    return jax.lax.map(f, tuple(xs), batch_size=batch_size).reshape((
        *batch,
        num_triangles,
    ))


@eqx.filter_jit
def first_triangles_hit_by_rays(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch num_triangles 3 3"],
    active_triangles: Bool[ArrayLike, "*#batch num_triangles"] | None = None,
    batch_size: int | None = 4096,
    ray_batch_size: int | None = None,
    tri_batch_size: int | None = None,
    **kwargs: Any,
) -> tuple[Int[Array, " *batch"], Float[Array, " *batch"]]:
    """
    Return the first triangle hit by each ray.

    This function should be used when allocating an array of size
    ``*batch num_triangles 3`` (or bigger) is not possible, and you are only interested in
    getting the first triangle hit by the ray.

    If two or more triangles are hit at the same distance, the one with the closest center to the ray origin is selected. Two triangles are considered to be hit at the same distance if their distances differ by less than ``100 * eps``, where ``eps`` is the machine epsilon of the floating point dtype.

    Args:
        ray_origins: An array of origin vertices.
        ray_directions: An array of ray direction. The ray ends
            should be equal to ``ray_origins + ray_directions``.
        triangle_vertices: An array of triangle vertices.
        active_triangles: An optional array of boolean values indicating
            which triangles are active, i.e., should be considered for intersection.

            If not specified, all triangles are considered active.
        batch_size: The default batch size used when either ``ray_batch_size`` or
            ``tri_batch_size`` is not specified. This allows to make a trade-off between memory
            usage and performance.

            If :data:`None`, the provided ``ray_batch_size`` and ``tri_batch_size`` values are
            used. Otherwise, this value is used as the default for whichever of
            ``ray_batch_size`` and ``tri_batch_size`` are left unspecified.
        ray_batch_size: The number of rays to process in a single batch.
            This allows to chunk rays and reduce peak memory usage.

            If :data:`None`, it defaults to ``batch_size``.
        tri_batch_size: The number of triangles to process in a single batch.
            This allows to chunk triangles and reduce peak memory usage.

            If :data:`None`, the triangle batch size defaults to ``batch_size``.
        kwargs: Keyword arguments passed to
            :func:`rays_intersect_triangles`.

    Returns:
        For each ray, return the index and to distance to the first triangle hit.

        If no triangle is hit, the index is set to ``-1`` and
        the distance is set to :data:`inf<numpy.inf>`.
    """
    ray_origins = jnp.asarray(ray_origins)
    ray_directions = jnp.asarray(ray_directions)
    triangle_vertices = jnp.asarray(triangle_vertices)

    if active_triangles is not None:
        active_triangles = jnp.asarray(active_triangles)

    dtype = jnp.result_type(ray_origins, ray_directions, triangle_vertices)
    dist_tol = jnp.asarray(100 * jnp.finfo(dtype).eps)

    num_triangles = triangle_vertices.shape[-3]

    batch = jnp.broadcast_shapes(
        ray_origins.shape[:-1],
        ray_directions.shape[:-1],
        triangle_vertices.shape[:-3],
        active_triangles.shape[:-1] if active_triangles is not None else (),
    )
    num_rays = math.prod(batch)
    ray_batch_size = (
        ray_batch_size
        if ray_batch_size is not None
        else batch_size
        if batch_size is not None
        else num_rays
    )
    ray_batch_size = min(ray_batch_size, num_rays)
    tri_batch_size = (
        tri_batch_size
        if tri_batch_size is not None
        else batch_size
        if batch_size is not None
        else num_triangles
    )
    tri_batch_size = min(tri_batch_size, num_triangles)
    if num_triangles == 0:
        # If there are no triangles, there are no hits
        return (
            jnp.full(batch, -1, dtype=jnp.int32),
            jnp.full(
                batch,
                jnp.inf,
                dtype=jnp.result_type(ray_origins, ray_directions, triangle_vertices),
            ),
        )

    xs = []
    argnames = []
    map_fn = partial(
        _first_triangle_hit_by_ray,
        batch_size=tri_batch_size,
        dist_tol=dist_tol,
        **kwargs,
    )
    if math.prod(ray_origins.shape[:-1]) > 1:
        xs.append(ray_origins.reshape(-1, 3))
        argnames.append("ray_origin")
    else:
        map_fn = partial(map_fn, ray_origin=ray_origins.reshape(3))
    if math.prod(ray_directions.shape[:-1]) > 1:
        xs.append(ray_directions.reshape(-1, 3))
        argnames.append("ray_direction")
    else:
        map_fn = partial(map_fn, ray_direction=ray_directions.reshape(3))
    if math.prod(triangle_vertices.shape[:-3]) > 1:
        xs.append(triangle_vertices.reshape(-1, num_triangles, 3, 3))
        argnames.append("triangle_vertices")
    else:
        map_fn = partial(
            map_fn, triangle_vertices=triangle_vertices.reshape(num_triangles, 3, 3)
        )
    if active_triangles is not None and math.prod(active_triangles.shape[:-1]) > 1:
        xs.append(active_triangles.reshape(-1, num_triangles))
        argnames.append("active_triangles")
    else:
        map_fn = partial(
            map_fn,
            active_triangles=active_triangles.reshape(num_triangles)
            if active_triangles is not None
            else None,
        )

    def f(args):
        return map_fn(**dict(zip(argnames, args, strict=True)))

    if not xs:
        indices, ts = _first_triangle_hit_by_ray(
            ray_origins.reshape(3),
            ray_directions.reshape(3),
            triangle_vertices.reshape(num_triangles, 3, 3),
            active_triangles.reshape(num_triangles)
            if active_triangles is not None
            else None,
            batch_size=tri_batch_size,
            dist_tol=dist_tol,
            **kwargs,
        )
        return indices.reshape(batch), ts.reshape(batch)

    if num_rays > ray_batch_size:
        indices, ts = jax.lax.map(f, tuple(xs), batch_size=ray_batch_size)
    else:
        indices, ts = jax.vmap(f)(jnp.stack(xs, axis=1))

    return indices.reshape(batch), ts.reshape(batch)
