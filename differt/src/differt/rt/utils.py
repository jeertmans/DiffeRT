"""
Ray Tracing utilities.

To generate a subset of all paths between two vertices, e.g.,
a transmitter TX and a received RX, methods generate each
path from a corresponding path candidate.

A path candidate is simply a list of primitive indices
to indicate with what primitive the path interact, and
in what order. The latter indicates that any permutation
of a given path candidate will result in another path.

I.e., the path candidate ``[4, 7]`` indicates that
the path first interacts with primitive ``4``, then
primitive ``7``, while the path candidate ``[7, 4]``
indicates a path interacting first with ``7`` then
with ``4``.

An empty path candidate indicates a direct path from
TX or RX, also known as line of sight path.

In general, interaction can be anything of the following:
reflection, diffraction, refraction, etc. The utilities
present in this module do not take care of the interaction type.

You can read more about path candidates in :cite:`mpt-eucap2023`.
"""
# ruff: noqa: ERA001

from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, ArrayLike, Bool, Float, Int, jaxtyped

from differt.geometry.utils import fibonacci_lattice
from differt_core.rt.graph import CompleteGraph

T = TypeVar("T")


class _SizedIterator(Generic[T]):
    """A custom generatic class that is both :class:`Iterator<collections.abc.Iterator>` and :class:`Sized<collections.abc.Sized>`.

    Args:
        iter_: The iterator.
        size: The size, i.e., length, of the iterator, or a callable that returns its current length.

    Examples:
        The following example shows how to create a sized iterator.

        >>> from differt.rt.utils import _SizedIterator
        >>> l = [1, 2, 3, 4, 5]
        >>> it = _SizedIterator(iter_=iter(l), size=5)
        >>> len(it)
        5
        >>> it = _SizedIterator(iter_=iter(l), size=l.__len__)
        >>> len(it)
        5

    """

    def __init__(self, iter_: Iterator[T], size: int | Callable[[], int]) -> None:
        self.iter_ = iter_
        self.size = size

    def __iter__(self) -> "_SizedIterator[T]":
        return self

    def __next__(self) -> T:
        return next(self.iter_)

    def __len__(self) -> int:
        if isinstance(self.size, int):
            return self.size
        return self.size()


@jaxtyped(typechecker=typechecker)
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


@jaxtyped(typechecker=typechecker)
def generate_all_path_candidates_iter(
    num_primitives: int,
    order: int,
) -> _SizedIterator[Int[Array, " order"]]:
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
    return _SizedIterator(m, size=it.__len__)


@jaxtyped(typechecker=typechecker)
def generate_all_path_candidates_chunks_iter(
    num_primitives: int,
    order: int,
    chunk_size: int = 1000,
) -> _SizedIterator[Int[Array, "chunk_size order"]]:
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
    return _SizedIterator(m, size=it.__len__)


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def rays_intersect_triangles(
    ray_origins: Float[Array, "*#batch 3"],
    ray_directions: Float[Array, "*#batch 3"],
    triangle_vertices: Float[Array, "*#batch 3 3"],
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
            split into multiple triangles.

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
            >>> from differt.geometry.utils import fibonacci_lattice
            >>> from differt.plotting import draw_rays
            >>> from differt.rt.utils import (
            ...     rays_intersect_triangles,
            ... )
            >>> from differt.scene.sionna import get_sionna_scene, download_sionna_scenes
            >>> from differt.scene.triangle_scene import TriangleScene
            >>>
            >>> download_sionna_scenes()
            >>> file = get_sionna_scene("simple_street_canyon")
            >>> scene = TriangleScene.load_xml(file)
            >>> scene = eqx.tree_at(lambda s: s.transmitters, scene, jnp.array([-33, 0, 32.0]))
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
            >>> ray_directions *= np.max(
            ...     t, axis=1, keepdims=True, initial=1.0, where=hit
            ... )  # Scale rays length before plotting
            >>> fig = draw_rays(  # We only plot rays hitting at least one triangle
            ...     np.asarray(ray_origins[rays_hit, :]),
            ...     np.asarray(ray_directions[rays_hit, :]),
            ...     backend="plotly",
            ...     line={"color": "red"},
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
    a = jnp.sum(h * edge_1, axis=-1)

    cond_a = jnp.abs(a) < epsilon

    f = jnp.where(a == 0.0, 0, 1.0 / a)
    s = ray_origins - vertex_0
    u = f * jnp.sum(s * h, axis=-1)

    cond_u = (u < 0.0) | (u > 1.0)

    q = jnp.cross(s, edge_1)
    v = f * jnp.sum(q * ray_directions, axis=-1)

    cond_v = (v < 0.0) | (u + v > 1.0)

    t = f * jnp.sum(q * edge_2, axis=-1)

    cond_t = t <= epsilon

    hit = ~(cond_a | cond_u | cond_v | cond_t)

    return t, hit


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def rays_intersect_any_triangle(
    ray_origins: Float[Array, "*#batch 3"],
    ray_directions: Float[Array, "*#batch 3"],
    triangle_vertices: Float[Array, "*#batch num_triangles 3 3"],
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
    ``t < hit_threshold & hit`` evaluates to :data:`True`.

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
    if hit_tol is None:
        dtype = jnp.result_type(ray_origins, ray_directions, triangle_vertices)
        hit_tol = 10.0 * jnp.finfo(dtype).eps

    hit_threshold = 1.0 - hit_tol  # type: ignore[reportOperatorIssue]

    # Put 'num_triangles' axis as leading axis
    triangle_vertices = jnp.moveaxis(triangle_vertices, -3, 0)

    batch = jnp.broadcast_shapes(
        ray_origins.shape[:-1],
        ray_directions.shape[:-1],
        triangle_vertices.shape[1:-2],
    )

    @jaxtyped(typechecker=typechecker)
    def scan_fun(
        intersect: Bool[Array, " *#batch"],
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
@jaxtyped(typechecker=typechecker)
def triangles_visible_from_vertices(
    vertices: Float[Array, "*#batch 3"],
    triangle_vertices: Float[Array, "*#batch num_triangles 3 3"],
    num_rays: int = int(1e6),
    **kwargs: Any,
) -> Bool[Array, "*batch num_triangles"]:
    """
    Return whether triangles are visible from vertex positions.

    This function uses ray launching and
    :func:`fibonacci_lattice<differt.geometry.utils.fibonacci_lattice>` to estimate
    whether a given triangle can be reached from a specific vertex, i.e., with a ray path,
    without interacting with any other triangle facet.

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
            >>> from differt.rt.utils import (
            ...     triangles_visible_from_vertices,
            ... )
            >>> from differt.scene.sionna import get_sionna_scene, download_sionna_scenes
            >>> from differt.scene.triangle_scene import TriangleScene
            >>>
            >>> download_sionna_scenes()
            >>> file = get_sionna_scene("simple_street_canyon")
            >>> scene = TriangleScene.load_xml(file)
            >>> scene = eqx.tree_at(lambda s: s.transmitters, scene, jnp.array([-33, 0, 32.0]))
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
    # [*batch 3]
    ray_origins = vertices

    # [num_rays 3]
    ray_directions = fibonacci_lattice(num_rays)

    batch = jnp.broadcast_shapes(ray_origins.shape[:-1], triangle_vertices.shape[:-3])

    @jaxtyped(typechecker=typechecker)
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
