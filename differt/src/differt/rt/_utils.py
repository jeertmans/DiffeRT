# ruff: noqa: ERA001

from collections.abc import Callable, Iterator, Sized
from functools import cache
from typing import TYPE_CHECKING, Any, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from differt.geometry import fibonacci_lattice, viewing_frustum
from differt.utils import dot
from differt_core.rt import CompleteGraph

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self
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
    with containing loops, i.e., two or more consecutive indices that are equal.

    Args:
        num_primitives: The (positive) number of primitives.
        order: The path order. An order less than one returns an empty array.

    Returns:
        An unsigned array with primitive indices on each columns. Its number of
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


@eqx.filter_jit
def rays_intersect_triangles(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch 3 3"],
    *,
    epsilon: Float[ArrayLike, " "] | None = None,
) -> tuple[Float[Array, "*batch"], Bool[Array, "*batch"]]:
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
            split into multiple triangles. shape

            If not specified, the default is ten times the epsilon value
            of the currently used floating point dtype.

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
            ...     jnp.asarray(ray_origins[rays_hit, :]),
            ...     jnp.asarray(ray_directions[rays_hit, :]),
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
    a = dot(h, edge_1)

    hit = jnp.abs(a) > epsilon

    f = jnp.where(a == 0.0, 0, 1.0 / a)
    s = ray_origins - vertex_0
    u = f * dot(s, h)

    hit &= (u >= 0.0) & (u <= 1.0)

    q = jnp.cross(s, edge_1)
    v = f * dot(q, ray_directions)

    hit &= (v >= 0.0) & (u + v <= 1.0)

    t = f * dot(q, edge_2)

    hit &= t > epsilon

    return t, hit


@eqx.filter_jit
def rays_intersect_any_triangle(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch num_triangles 3 3"],
    *,
    hit_tol: Float[ArrayLike, " "] | None = None,
    **kwargs: Any,
) -> Bool[Array, " *batch"]:
    """
    Return whether rays intersect any of the triangles using the Möller-Trumbore algorithm.

    This function should be used when allocating an array of size
    ``*batch num_triangles 3`` (or bigger) is not possible, and you are only interested in
    checking if at least one of the triangles is intersect.

    A triangle is considered to be intersected if
    ``t < (1 - hit_tol) & hit`` evaluates to :data:`True`.

    Args:
        ray_origins: An array of origin vertices.
        ray_directions: An array of ray direction. The ray ends
            should be equal to ``ray_origins + ray_directions``.
        triangle_vertices: An array of triangle vertices.
        hit_tol: The tolerance applied to check if a ray hits another object or not,
            before it reaches the expected position, i.e., the 'interaction' object.

            Using a non-zero tolerance is required as it would otherwise trigger
            false positives.

            If not specified, the default is ten times the epsilon value
            of the currently used floating point dtype.
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

    # Put 'num_triangles' axis as leading axis
    triangle_vertices = jnp.moveaxis(triangle_vertices, -3, 0)

    batch = jnp.broadcast_shapes(
        ray_origins.shape[:-1],
        ray_directions.shape[:-1],
        triangle_vertices.shape[1:-2],
    )

    def scan_fun(
        intersect: Bool[Array, " *batch"],
        triangle_vertices: Float[Array, "*#batch 3 3"],
    ) -> tuple[Bool[Array, " *batch"], None]:
        t, hit = rays_intersect_triangles(
            ray_origins,
            ray_directions,
            triangle_vertices,
            **kwargs,
        )
        intersect = intersect | ((t < hit_threshold) & hit)
        return intersect, None

    return jax.lax.scan(
        scan_fun,
        init=jnp.zeros(batch, dtype=bool),
        xs=triangle_vertices,
    )[0]


@eqx.filter_jit
def triangles_visible_from_vertices(
    vertices: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch num_triangles 3 3"],
    num_rays: int = int(1e6),
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
        num_rays: The number of rays to launch.

            The larger, the more accurate.
        kwargs: Keyword arguments passed to
            :func:`rays_intersect_triangles`.

    Returns:
        For each triangle, whether it intersects with any of the rays.

    Examples:
        The following example shows how to identify triangles as
        visible from a given transmitter, coloring them in dark gray.

        .. plotly::

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
    """
    vertices = jnp.asarray(vertices)
    triangle_vertices = jnp.asarray(triangle_vertices)

    # [*batch 3]
    ray_origins = vertices

    # [2 3]
    # note: currently we don't handle batches and generate one frustum for everything
    frustum = viewing_frustum(
        ray_origins,
        triangle_vertices.reshape(*triangle_vertices.shape[:-3], -1, 3),
        reduce=True,
    )

    # [num_rays 3]
    ray_directions = fibonacci_lattice(num_rays, frustum=frustum)

    batch = jnp.broadcast_shapes(ray_origins.shape[:-1], triangle_vertices.shape[:-3])

    def scan_fun(
        visible: Bool[Array, "*batch num_triangles"],
        ray_direction: Float[Array, "3"],
    ) -> tuple[Bool[Array, " *batch num_triangles"], None]:
        t, hit = rays_intersect_triangles(
            ray_origins[..., None, :],
            ray_direction[..., None, :],
            triangle_vertices,
            **kwargs,
        )
        # A triangle is visible if it is the first triangle to be intersected by a ray.
        visible = visible | (
            t == jnp.min(t, axis=-1, keepdims=True, initial=jnp.inf, where=hit)
        )

        return visible, None

    return jax.lax.scan(
        scan_fun,
        init=jnp.zeros((*batch, triangle_vertices.shape[-3]), dtype=bool),
        xs=ray_directions,
    )[0]


@eqx.filter_jit
def first_triangles_hit_by_rays(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch num_triangles 3 3"],
    **kwargs: Any,
) -> tuple[Int[Array, " *batch"], Float[Array, " *batch"]]:
    """
    Return the first triangle hit by each ray.

    This function should be used when allocating an array of size
    ``*batch num_triangles 3`` (or bigger) is not possible, and you are only interested in
    getting the first triangle hit by the ray.

    Args:
        ray_origins: An array of origin vertices.
        ray_directions: An array of ray direction. The ray ends
            should be equal to ``ray_origins + ray_directions``.
        triangle_vertices: An array of triangle vertices.
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

    # Put 'num_triangles' axis as leading axis
    triangle_vertices = jnp.moveaxis(triangle_vertices, -3, 0)

    batch = jnp.broadcast_shapes(
        ray_origins.shape[:-1],
        ray_directions.shape[:-1],
        triangle_vertices.shape[1:-2],
    )

    def scan_fun(
        carry: tuple[Int[Array, " *batch"], Float[Array, "*batch"], Int[Array, " "]],
        triangle_vertices: Float[Array, "*#batch 3 3"],
    ) -> tuple[
        tuple[Int[Array, " *batch"], Float[Array, "*batch"], Int[Array, " "]], None
    ]:
        indices, t_hit, index = carry
        t, hit = rays_intersect_triangles(
            ray_origins,
            ray_directions,
            triangle_vertices,
            **kwargs,
        )
        t = jnp.where(hit, t, jnp.inf)
        indices = jnp.where(t < t_hit, index, indices)
        t_hit = jnp.minimum(t, t_hit)
        return (indices, t_hit, index + 1), None

    return jax.lax.scan(
        scan_fun,
        init=(-jnp.ones(batch, dtype=int), jnp.full(batch, jnp.inf), jnp.array(0)),
        xs=triangle_vertices,
    )[0][:2]
