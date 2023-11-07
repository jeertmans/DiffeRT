from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, UInt, jaxtyped
from typeguard import typechecked as typechecker


@jaxtyped
@typechecker
@jax.jit
def _fill_path_candidates(path_candidates: UInt[Array, "num_primitives num_candidates_per_primitive order"]) -> UInt[Array, "num_primitives*num_candidates_per_primitive order"]:
    num_primitives, num_candidates_per_primitive, order = path_candidates.shape
    num_candidates = num_primitives * num_candidates_per_primitive
    batch_size = num_candidates_per_primitive

    path_candidates = jnp.reshape(path_candidates, (num_candidates, order))

    fill_value = 0
    for j in range(order):
        for i in range(0, num_candidates, batch_size):
            fill_value = jnp.where(
                jnp.logical_and(j > 0, fill_value == path_candidates[i, j - 1]),
                (fill_value + 1) % num_primitives,
                fill_value,
            )

            path_candidates = path_candidates.at[i : i + batch_size, j].set(fill_value)
            fill_value = (fill_value + 1) % num_primitives

        batch_size = batch_size // (num_primitives - 1)

    return path_candidates


@jaxtyped
@typechecker
def generate_path_candidates(
    num_primitives: int, order: int
) -> UInt[Array, "num_candidates order"]:
    """
    Generate an array of path candidates for fixed path order
    and a number of primitives.

    The returned array contains, for each row, an array of
    ``order`` indices indicating the primitive with which the path interacts.

    This list is generated as the list of all simple paths from one node to
    another, by passing by exactly ``order`` primitives. Calling this function
    is equivalent to calling :func:`itertools.product` with parameters
    ``[0, 1, ..., num_primitives - 1]`` and ``repeat=order``, and removing entries
    with containing loops, i.e., two or more consecutive indices that are equal.

    Args:
        num_primitives: the (positive) number of primitives.
        order: the path order. An order less than one returns an empty array.

    Returns:
        An unsigned array with primitive indices on each row. Its number of rows
        is actually equal to
        ``num_primitives * ((num_primitives - 1) ** (order - 1))``.
    """
    if order < 1:
        return jnp.empty((0, 0), dtype=jnp.uint32)
    elif order == 1:
        indices = jnp.arange(num_primitives, dtype=jnp.uint32)
        return jnp.reshape(indices, (-1, 1))

    num_candidates_per_primitive = ((num_primitives - 1) ** (order - 1))
    path_candidates = jnp.empty((num_primitives, num_candidates_per_primitive, order), dtype=jnp.uint32)

    return _fill_path_candidates(path_candidates)


@jaxtyped
@typechecker
def rays_intersect_triangles(
    ray_origins: Float[Array, "*batch 3"],
    ray_directions: Float[Array, "*batch 3"],
    triangle_vertices: Float[Array, "*batch 3 3"],
) -> tuple[Float[Array, " *batch"], Bool[Array, " *batch"]]:
    """
    Return whether rays intersect corresponding triangles using the
    Möller-Trumbore algorithm.

    The current implementation closely follows the C++ code from Wikipedia.
    """
    epsilon = 1e-07

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
