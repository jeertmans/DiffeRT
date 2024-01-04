"""Ray Tracing utilies."""

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Scalar, UInt, jaxtyped
from typeguard import typechecked as typechecker

from .. import _core


@jaxtyped(typechecker=typechecker)
def generate_all_path_candidates(
    num_primitives: int, order: int
) -> UInt[Array, "order num_candidates"]:
    """
    Generate an array of all path candidates for fixed path order
    and a number of primitives.

    The returned array contains, for each column, an array of
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
        _core.rt.utils.generate_all_path_candidates(num_primitives, order),
        dtype=jnp.uint32,
    )


@jaxtyped(typechecker=typechecker)
def rays_intersect_triangles(
    ray_origins: Float[Array, "*batch 3"],
    ray_directions: Float[Array, "*batch 3"],
    triangle_vertices: Float[Array, "*batch 3 3"],
    epsilon: Float[Scalar, ""] = 1e-6,
) -> tuple[Float[Array, " *batch"], Bool[Array, " *batch"]]:
    """
    Return whether rays intersect corresponding triangles using the
    Möller-Trumbore algorithm.

    The current implementation closely follows the C++ code from Wikipedia.

    Args:
        ray_origins: An array of origin vertices.
        ray_directions: An array of ray direction. The ray ends
            should be equal to ``ray_origins + ray_directions``.
        triangle_vertices: An array of triangle vertices.
        epsilon: A small tolerance threshold that allows rays
            to hit the triangles slightly outside the actual area.
            A positive value virtually increases the size of the triangles,
            a negative value will have the opposite effect.

            Such a tolerance is especially useful when rays are hitting
            triangle edges, a very common case if geometries are planes
            split into multiple triangles.

    Returns:
        For each ray, return the scale factor of ``ray_directions`` for the
        vector to reach the corresponding triangle, and whether the intersection
        actually lies inside the triangle.
    """
    vertex_0 = triangle_vertices[..., 0, :]
    vertex_1 = triangle_vertices[..., 1, :]
    vertex_2 = triangle_vertices[..., 2, :]

    edge_1 = vertex_1 - vertex_0
    edge_2 = vertex_2 - vertex_0

    h = jnp.cross(ray_directions, edge_2, axis=-1)
    a = jnp.sum(edge_1 * h, axis=-1)

    cond_a = (a > -epsilon) & (a < epsilon)

    f = 1.0 / a
    s = ray_origins - vertex_0
    u = f * jnp.sum(s * h, axis=-1)

    cond_u = (u < 0.0) | (u > 1.0)

    q = jnp.cross(s, edge_1, axis=-1)
    v = f * jnp.sum(ray_directions * q, axis=-1)

    cond_v = (v < 0.0) | (u + v > 1.0)

    t = f * jnp.sum(edge_2 * q, axis=-1)

    cond_t = t <= epsilon

    return t, ~(cond_a | cond_u | cond_v | cond_t)
