# ruff: noqa: ERA001
import typing
from collections.abc import Callable, Iterator, Sized
from functools import cache
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
    epsilon: Float[ArrayLike, " "] | None = None,
    smoothing_factor: Float[ArrayLike, " "],
) -> tuple[Float[Array, " *batch"], Bool[Array, " *batch"]]: ...


@overload
def rays_intersect_triangles(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch 3 3"],
    *,
    epsilon: Float[ArrayLike, " "] | None = None,
    smoothing_factor: None = None,
) -> tuple[Float[Array, " *batch"], Float[Array, " *batch"]]: ...


@eqx.filter_jit
def rays_intersect_triangles(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch 3 3"],
    *,
    epsilon: Float[ArrayLike, " "] | None = None,
    smoothing_factor: Float[ArrayLike, " "] | None = None,
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


@overload
def rays_intersect_any_triangle(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch num_triangles 3 3"],
    active_triangles: Bool[ArrayLike, "*#batch num_triangles"] | None = None,
    *,
    hit_tol: Float[ArrayLike, " "] | None = None,
    smoothing_factor: None = None,
    batch_size: int | None = 512,
    **kwargs: Any,
) -> Bool[Array, " *batch"]: ...


@overload
def rays_intersect_any_triangle(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch num_triangles 3 3"],
    active_triangles: Bool[ArrayLike, "*#batch num_triangles"] | None = None,
    *,
    hit_tol: Float[ArrayLike, " "] | None = None,
    smoothing_factor: Float[ArrayLike, " "],
    batch_size: int | None = 512,
    **kwargs: Any,
) -> Float[Array, " *batch"]: ...


@eqx.filter_jit
def rays_intersect_any_triangle(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch num_triangles 3 3"],
    active_triangles: Bool[ArrayLike, "*#batch num_triangles"] | None = None,
    *,
    hit_tol: Float[ArrayLike, " "] | None = None,
    smoothing_factor: Float[ArrayLike, " "] | None = None,
    batch_size: int | None = 512,
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
        batch_size: The number of triangles to process in a single batch.
            This allows to make a trade-off between memory usage and performance.

            The batch size is automatically adjusted to be the minimum of the number of triangles
            and the specified batch size.

            If :data:`None`, the batch size is set to the number of triangles.
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
    if batch_size is None:
        batch_size = num_triangles
    batch_size = max(min(batch_size, num_triangles), 1)
    num_batches, rem = divmod(num_triangles, batch_size)

    if active_triangles is not None:
        active_triangles = jnp.asarray(active_triangles)

    # Combine the batch dimensions
    batch = jnp.broadcast_shapes(
        ray_origins.shape[:-1],
        ray_directions.shape[:-1],
        triangle_vertices.shape[:-3],
        active_triangles.shape[:-1] if active_triangles is not None else (),
    )

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

    def map_fn(
        ray_origins: Float[Array, "*#batch 3"],
        ray_directions: Float[Array, "*#batch 3"],
        triangle_vertices: Float[Array, "*#batch num_triangles 3 3"],
        active_triangles: Bool[Array, "*#batch num_triangles"] | None = None,
    ) -> Bool[Array, " *batch"] | Float[Array, " *batch"]:
        t, hit = rays_intersect_triangles(
            ray_origins[..., None, :],
            ray_directions[..., None, :],
            triangle_vertices,
            smoothing_factor=smoothing_factor,
            **kwargs,
        )
        if smoothing_factor is not None:
            return jnp.minimum(
                hit, smoothing_function(hit_threshold - t, smoothing_factor)
            ).sum(axis=-1, where=active_triangles)
        return ((t < hit_threshold) & hit).any(axis=-1, where=active_triangles)

    def reduce_fn(
        left: Bool[Array, " *batch"] | Float[Array, " *batch"],
        right: Bool[Array, " *batch"] | Float[Array, " *batch"],
    ) -> Bool[Array, " *batch"] | Float[Array, " *batch"]:
        if smoothing_factor is not None:
            return (left + right).clip(max=1.0)
        return left | right

    def body_fun(
        batch_index: Int[Array, " "],
        intersect: Bool[Array, " *batch"] | Float[Array, " *batch"],
    ) -> Bool[Array, " *batch"] | Float[Array, " *batch"]:
        start_index = batch_index * batch_size
        batch_of_triangle_vertices = jax.lax.dynamic_slice_in_dim(
            triangle_vertices, start_index, batch_size, axis=-3
        )
        batch_of_active_triangles = (
            jax.lax.dynamic_slice_in_dim(
                active_triangles, start_index, batch_size, axis=-1
            )
            if active_triangles is not None
            else None
        )
        return reduce_fn(
            intersect,
            map_fn(
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                triangle_vertices=batch_of_triangle_vertices,
                active_triangles=batch_of_active_triangles,
            ),
        )

    init_val = (
        jnp.zeros(batch)
        if smoothing_factor is not None
        else jnp.zeros(batch, dtype=jnp.bool)
    )

    intersect = jax.lax.fori_loop(
        0,
        num_batches,
        body_fun,
        init_val=init_val,
    )

    if rem > 0:
        return reduce_fn(
            intersect,
            map_fn(
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                triangle_vertices=triangle_vertices[..., -rem:, :, :],
                active_triangles=active_triangles[..., -rem:]
                if active_triangles is not None
                else None,
            ),
        )
    return intersect


@eqx.filter_jit
def triangles_visible_from_vertices(
    vertices: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch num_triangles 3 3"],
    active_triangles: Bool[ArrayLike, "*#batch num_triangles"] | None = None,
    num_rays: int = int(1e6),
    batch_size: int | None = 512,
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
        batch_size: The number of rays to process in a single batch.
            This allows to make a trade-off between memory usage and performance.

            The batch size is automatically adjusted to be the minimum of the number of rays
            and the specified batch size.

            If :data:`None`, the batch size is set to the number of rays.
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
    triangle_centers = triangle_vertices.mean(axis=-2, keepdims=True)
    world_vertices = jnp.concat((triangle_vertices, triangle_centers), axis=-2).reshape(
        *triangle_vertices.shape[:-3], -1, 3
    )

    if active_triangles is not None:
        active_triangles = jnp.asarray(active_triangles)
        active_vertices = jnp.repeat(active_triangles, 4, axis=-1)
    else:
        active_vertices = None

    # [*batch 3]
    ray_origins = vertices

    # [*batch 2 3]
    frustum = viewing_frustum(
        ray_origins,
        world_vertices,
        active_vertices=active_vertices,
    )

    batch_size = num_rays if batch_size is None else min(batch_size, num_rays)
    num_batches, rem = divmod(num_rays, batch_size)

    # [*batch num_rays 3]
    ray_directions = jnp.vectorize(
        lambda n, frustum: fibonacci_lattice(n, frustum=frustum),
        excluded={0},
        signature="(2,3)->(n,3)",
    )(num_rays, frustum)

    # Combine the batch dimensions
    batch = jnp.broadcast_shapes(
        ray_origins.shape[:-2],
        ray_directions.shape[:-2],
        triangle_vertices.shape[:-3],
        active_triangles.shape[:-1] if active_triangles is not None else (),
    )

    def update_visible_triangles(
        visible_triangles: Bool[Array, "*#batch num_triangles"],
        visible_indices: Int[Array, "*batch batch_size"],
    ) -> Bool[Array, "*#batch num_triangles"]:
        indices = jnp.indices(visible_triangles.shape, sparse=True)
        indices = (*indices[:-1], visible_indices)
        return visible_triangles.at[indices].set(True, wrap_negative_indices=False)

    def map_fn(
        ray_origins: Float[Array, "*#batch 3"],
        ray_directions: Float[Array, "*#batch batch_size 3"],
        triangle_vertices: Float[Array, "*#batch num_triangles 3 3"],
        active_triangles: Bool[Array, "*#batch num_triangles"] | None = None,
    ) -> Int[Array, "*batch batch_size"]:
        indices, _ = first_triangles_hit_by_rays(
            ray_origins[..., None, :],
            ray_directions,
            triangle_vertices[..., None, :, :, :],
            active_triangles=active_triangles[..., None, :]
            if active_triangles is not None
            else None,
            batch_size=None,
            **kwargs,
        )
        return indices

    def body_fun(
        batch_index: Int[Array, " "],
        visible_triangles: Bool[Array, "*batch num_triangles"],
    ) -> Bool[Array, "*batch num_triangles"]:
        start_index = batch_index * batch_size
        batch_of_ray_directions = jax.lax.dynamic_slice_in_dim(
            ray_directions, start_index, batch_size, axis=-2
        )
        visible_indices = map_fn(
            ray_origins,
            batch_of_ray_directions,
            triangle_vertices,
            active_triangles,
        )
        return update_visible_triangles(visible_triangles, visible_indices)

    init_val = jnp.zeros((*batch, triangle_vertices.shape[-3]), dtype=jnp.bool)

    visible_triangles = jax.lax.fori_loop(
        0,
        num_batches,
        body_fun,
        init_val=init_val,
    )

    if rem > 0:
        visible_indices = map_fn(
            ray_origins,
            ray_directions[..., -rem:, :],
            triangle_vertices,
            active_triangles,
        )
        return update_visible_triangles(visible_triangles, visible_indices)
    return visible_triangles


@eqx.filter_jit
def first_triangles_hit_by_rays(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch num_triangles 3 3"],
    active_triangles: Bool[ArrayLike, "*#batch num_triangles"] | None = None,
    batch_size: int | None = 512,
    **kwargs: Any,
) -> tuple[Int[Array, " *batch"], Float[Array, " *batch"]]:
    """
    Return the first triangle hit by each ray.

    This function should be used when allocating an array of size
    ``*batch num_triangles 3`` (or bigger) is not possible, and you are only interested in
    getting the first triangle hit by the ray.

    If two or more triangles are hit at the same distance, the one with the closest center to the ray origin is selected. Two triangles are considered to be hit at the same distance if their distances differ by less than ``100 * eps``, or ten times the ``epsilon`` keyword argument passed to :func:`rays_intersect_triangles`.

    Args:
        ray_origins: An array of origin vertices.
        ray_directions: An array of ray direction. The ray ends
            should be equal to ``ray_origins + ray_directions``.
        triangle_vertices: An array of triangle vertices.
        active_triangles: An optional array of boolean values indicating
            which triangles are active, i.e., should be considered for intersection.

            If not specified, all triangles are considered active.
        batch_size: The number of triangles to process in a single batch.
            This allows to make a trade-off between memory usage and performance.

            The batch size is automatically adjusted to be the minimum of the number of triangles
            and the specified batch size.

            If :data:`None`, the batch size is set to the number of triangles.
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

    if epsilon := kwargs.get("epsilon"):
        epsilon = 10 * jnp.asarray(epsilon)
    else:
        dtype = jnp.result_type(ray_origins, ray_directions, triangle_vertices)
        epsilon = jnp.asarray(100 * jnp.finfo(dtype).eps)

    num_triangles = triangle_vertices.shape[-3]
    if batch_size is None:
        batch_size = num_triangles
    batch_size = max(min(batch_size, num_triangles), 1)
    num_batches, rem = divmod(num_triangles, batch_size)

    if active_triangles is not None:
        active_triangles = jnp.asarray(active_triangles)

    # Combine the batch dimensions
    batch = jnp.broadcast_shapes(
        ray_origins.shape[:-1],
        ray_directions.shape[:-1],
        triangle_vertices.shape[:-3],
        active_triangles.shape[:-1] if active_triangles is not None else (),
    )

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

    def reduce_fn(
        left: tuple[
            Int[Array, " *batch"],
            Float[Array, " *batch"],
            Float[Array, " *batch"],
            Float[Array, " *#batch"],
        ],
        right: tuple[
            Int[Array, " *batch"],
            Float[Array, " *batch"],
            Float[Array, " *batch"],
            Float[Array, " *#batch"],
        ],
    ) -> tuple[
        Int[Array, " *batch"],
        Float[Array, " *batch"],
        Float[Array, " *batch"],
        Float[Array, " *#batch"],
    ]:
        left_indices, left_t, left_center_distances, eps = left
        right_indices, right_t, right_center_distances, _ = right
        cond: Array = jnp.where(
            jnp.abs(left_t - right_t) < eps,
            left_center_distances < right_center_distances,
            left_t < right_t,
        )
        t = jnp.where(cond, left_t, right_t)
        indices = jnp.where(cond, left_indices, right_indices)
        t = jnp.minimum(left_t, right_t)
        center_distances = jnp.where(
            cond, left_center_distances, right_center_distances
        )
        is_finite = jnp.isfinite(t)
        indices = jnp.where(is_finite, indices, -1)
        t = jnp.where(is_finite, t, jnp.inf)
        return indices, t, center_distances, eps

    def map_fn(
        ray_origins: Float[Array, "*#batch 3"],
        ray_directions: Float[Array, "*#batch 3"],
        triangle_vertices: Float[Array, "*#batch num_triangles 3 3"],
        active_triangles: Bool[Array, "*#batch num_triangles"] | None = None,
    ) -> tuple[Int[Array, " *batch"], Float[Array, " *batch"], Float[Array, " *batch"]]:
        t, hit = rays_intersect_triangles(
            ray_origins[..., None, :],
            ray_directions[..., None, :],
            triangle_vertices,
            **kwargs,
        )
        if active_triangles is not None:
            hit &= active_triangles
        t = jnp.where(hit, t, jnp.inf)
        indices = jnp.arange(triangle_vertices.shape[-3])
        indices = jnp.broadcast_to(indices, t.shape)
        center_distances = jnp.linalg.norm(
            triangle_vertices.mean(axis=-2) - ray_origins[..., None, :], axis=-1
        )
        center_distances = jnp.broadcast_to(center_distances, t.shape)
        eps = jnp.broadcast_to(epsilon, t.shape)
        return jax.lax.reduce(
            (indices, t, center_distances, eps),
            (-1, jnp.inf, jnp.inf, epsilon),
            reduce_fn,
            dimensions=(t.ndim - 1,),
        )[:3]

    def body_fun(
        batch_index: Int[Array, " "],
        carry: tuple[
            Int[Array, " *batch"], Float[Array, " *batch"], Float[Array, " *batch"]
        ],
    ) -> tuple[Int[Array, " *batch"], Float[Array, " *batch"], Float[Array, " *batch"]]:
        start_index = batch_index * batch_size
        batch_of_triangle_vertices = jax.lax.dynamic_slice_in_dim(
            triangle_vertices, start_index, batch_size, axis=-3
        )
        batch_of_active_triangles = (
            jax.lax.dynamic_slice_in_dim(
                active_triangles, start_index, batch_size, axis=-1
            )
            if active_triangles is not None
            else None
        )
        indices, t, center_distances = map_fn(
            ray_origins,
            ray_directions,
            batch_of_triangle_vertices,
            batch_of_active_triangles,
        )
        return reduce_fn(
            (*carry, epsilon),
            (indices + start_index, t, center_distances, epsilon),
        )[:3]

    init_val = (
        -jnp.ones(batch, dtype=jnp.int32),
        jnp.full(
            batch,
            jnp.inf,
            dtype=jnp.result_type(ray_origins, ray_directions, triangle_vertices),
        ),
        jnp.full(
            batch,
            jnp.inf,
            dtype=jnp.result_type(ray_origins, ray_directions, triangle_vertices),
        ),
    )

    indices_and_t_and_center_distances = jax.lax.fori_loop(
        0,
        num_batches,
        body_fun,
        init_val=init_val,
    )

    if rem > 0:
        indices, t, center_distances = map_fn(
            ray_origins,
            ray_directions,
            triangle_vertices[..., -rem:, :, :],
            active_triangles[..., -rem:] if active_triangles is not None else None,
        )
        return reduce_fn(
            (*indices_and_t_and_center_distances, epsilon),
            (indices + num_batches * batch_size, t, center_distances, epsilon),
        )[:2]
    return indices_and_t_and_center_distances[:2]
