from typing import Any

import chex
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from differt.rt.utils import generate_all_path_candidates, rays_intersect_triangles
from differt.utils import sorted_array2


def uint_array(array_like: Any) -> Array:
    return jnp.array(array_like, dtype=jnp.uint32)


@pytest.mark.parametrize(
    "num_primitives,order,expected",
    [
        (0, 0, jnp.empty((0, 1), dtype=jnp.uint32)),
        (8, 0, jnp.empty((0, 1), dtype=jnp.uint32)),
        (0, 5, jnp.empty((5, 0), dtype=jnp.uint32)),
        (3, 1, uint_array([[0, 1, 2]])),
        (3, 2, uint_array([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]).T),
        (
            3,
            3,
            uint_array(
                [
                    [0, 1, 0],
                    [0, 1, 2],
                    [0, 2, 0],
                    [0, 2, 1],
                    [1, 0, 1],
                    [1, 0, 2],
                    [1, 2, 0],
                    [1, 2, 1],
                    [2, 0, 1],
                    [2, 0, 2],
                    [2, 1, 0],
                    [2, 1, 2],
                ]
            ).T,
        ),
    ],
)
def test_generate_all_path_candidates(
    num_primitives: int, order: int, expected: Array
) -> None:
    got = generate_all_path_candidates(num_primitives, order)
    got = sorted_array2(got.T).T  # order may not be the same so we sort
    chex.assert_trees_all_equal_shapes_and_dtypes(got, expected)
    chex.assert_trees_all_equal(got, expected)


@pytest.mark.parametrize(
    "ray_orig,ray_dest,expected",
    [
        (jnp.array([0.5, 0.5, 1.0]), jnp.array([0.5, 0.5, -1.0]), jnp.array(True)),
        (jnp.array([0.0, 0.0, 1.0]), jnp.array([1.0, 1.0, -1.0]), jnp.array(True)),
        (jnp.array([0.5, 0.5, 1.0]), jnp.array([0.5, 0.5, +0.5]), jnp.array(False)),
        (jnp.array([0.5, 0.5, 1.0]), jnp.array([1.0, 1.0, +1.0]), jnp.array(False)),
        (jnp.array([0.5, 0.5, 1.0]), jnp.array([1.0, 1.0, +1.5]), jnp.array(False)),
    ],
)
def test_rays_intersect_triangles(
    ray_orig: Array, ray_dest: Array, expected: Array
) -> None:
    triangle_vertices = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    t, hit = rays_intersect_triangles(
        ray_orig,
        ray_dest - ray_orig,
        triangle_vertices,
    )
    got = (t < 1.0) & hit
    chex.assert_trees_all_equal(got, expected)
