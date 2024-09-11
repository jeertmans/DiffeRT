from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import chex
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from differt.rt.utils import (
    generate_all_path_candidates,
    generate_all_path_candidates_chunks_iter,
    generate_all_path_candidates_iter,
    rays_intersect_any_triangle,
    rays_intersect_triangles,
)
from differt.utils import sorted_array2

from ..utils import random_inputs


@pytest.mark.parametrize(
    ("num_primitives", "order", "expected"),
    [
        (0, 0, jnp.empty((1, 0), dtype=int)),
        (8, 0, jnp.empty((1, 0), dtype=int)),
        (0, 5, jnp.empty((0, 5), dtype=int)),
        (3, 1, jnp.array([[0], [1], [2]])),
        (3, 2, jnp.array([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]])),
        (
            3,
            3,
            jnp.array(
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
                ],
            ),
        ),
    ],
)
def test_generate_all_path_candidates(
    num_primitives: int,
    order: int,
    expected: Array,
) -> None:
    got = generate_all_path_candidates(num_primitives, order)
    got = sorted_array2(got)  # order may not be the same so we sort
    chex.assert_trees_all_equal_shapes_and_dtypes(got, expected)
    chex.assert_trees_all_equal(got, expected)


@pytest.mark.parametrize(
    ("num_primitives", "order"),
    [
        (3, 1),
        (3, 2),
        (3, 3),
        (5, 4),
    ],
)
def test_generate_all_path_candidates_iter(num_primitives: int, order: int) -> None:
    expected = generate_all_path_candidates(num_primitives, order)
    expected = sorted_array2(expected)
    got = list(generate_all_path_candidates_iter(num_primitives, order))
    got = jnp.asarray(got)
    got = sorted_array2(got)

    chex.assert_trees_all_equal_shapes_and_dtypes(got, expected)
    chex.assert_trees_all_equal(got, expected)


@pytest.mark.parametrize(
    ("num_primitives", "order"),
    [
        (11, 1),
        (12, 3),
        (15, 4),
    ],
)
@pytest.mark.parametrize("chunk_size", [1, 10, 23])
def test_generate_all_path_candidates_chunks_iter(
    num_primitives: int, order: int, chunk_size: int
) -> None:
    it = generate_all_path_candidates_chunks_iter(num_primitives, order, chunk_size)

    previous_chunk = None

    try:
        while True:
            chunk = next(it)

            if previous_chunk is not None:
                chex.assert_shape(previous_chunk, (chunk_size, order))

            previous_chunk = chunk

    except StopIteration:
        pass

    if previous_chunk is not None:
        last_chunk_size, last_chunk_order = previous_chunk.shape
        assert last_chunk_size <= chunk_size
        assert last_chunk_order == order


@pytest.mark.parametrize(
    ("ray_orig", "ray_dest", "expected"),
    [
        (jnp.array([0.5, 0.5, 1.0]), jnp.array([0.5, 0.5, -1.0]), jnp.array(True)),
        (jnp.array([0.0, 0.0, 1.0]), jnp.array([1.0, 1.0, -1.0]), jnp.array(True)),
        (jnp.array([0.5, 0.5, 1.0]), jnp.array([0.5, 0.5, +0.5]), jnp.array(False)),
        (jnp.array([0.5, 0.5, 1.0]), jnp.array([1.0, 1.0, +1.0]), jnp.array(False)),
        (jnp.array([0.5, 0.5, 1.0]), jnp.array([1.0, 1.0, +1.5]), jnp.array(False)),
    ],
)
def test_rays_intersect_triangles(
    ray_orig: Array,
    ray_dest: Array,
    expected: Array,
) -> None:
    triangle_vertices = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    t, hit = rays_intersect_triangles(
        ray_orig,
        ray_dest - ray_orig,
        triangle_vertices,
    )
    got = (t < 1.0) & hit
    chex.assert_trees_all_equal(got, expected)


@pytest.mark.parametrize(
    ("ray_origins", "ray_directions", "triangle_vertices", "expectation"),
    [
        ((20, 10, 3), (20, 10, 3), (15, 3, 3), does_not_raise()),
        ((10, 3), (10, 3), (15, 3, 3), does_not_raise()),
        ((3,), (3,), (1, 3, 3), does_not_raise()),
        (
            (10, 3),
            (20, 3),
            (15, 3, 3),
            pytest.raises(TypeError),
        ),
        (
            (10, 3),
            (10, 4),
            (15, 3, 3),
            pytest.raises(TypeError),
        ),
    ],
)
@pytest.mark.parametrize("epsilon", [1e-6, 1e-2])
@pytest.mark.parametrize("hit_threshold", [1.0, 0.999, 1.5, 0.5])
@random_inputs("ray_origins", "ray_directions", "triangle_vertices")
def test_rays_intersect_any_triangle(
    ray_origins: Array,
    ray_directions: Array,
    triangle_vertices: Array,
    epsilon: float,
    hit_threshold: float,
    expectation: AbstractContextManager[Exception],
) -> None:
    with expectation:
        *batch, _ = ray_origins.shape
        num_triangles = triangle_vertices.shape[0]
        got = rays_intersect_any_triangle(
            ray_origins,
            ray_directions,
            triangle_vertices,
            epsilon=epsilon,
            hit_threshold=hit_threshold,
        )
        shape = (*batch, num_triangles, 3)
        ray_origins = jnp.broadcast_to(
            jnp.expand_dims(ray_origins, axis=-2),
            shape,
        )
        ray_directions = jnp.broadcast_to(
            jnp.expand_dims(ray_directions, axis=-2),
            shape,
        )
        triangle_vertices = jnp.broadcast_to(triangle_vertices, (*shape, 3))
        expected_t, expected_hit = rays_intersect_triangles(
            ray_origins,
            ray_directions,
            triangle_vertices,
            epsilon=epsilon,
        )
        expected = jnp.any((expected_t < hit_threshold) & expected_hit, axis=-1)

        chex.assert_trees_all_equal(got, expected)
