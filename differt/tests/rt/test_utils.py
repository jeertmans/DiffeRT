import sys
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import chex
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from differt.geometry import TriangleMesh
from differt.rt._utils import (
    first_triangles_hit_by_rays,
    generate_all_path_candidates,
    generate_all_path_candidates_chunks_iter,
    generate_all_path_candidates_iter,
    rays_intersect_any_triangle,
    rays_intersect_triangles,
    triangles_visible_from_vertices,
)
from differt.utils import sorted_array2

from ..utils import random_inputs


@pytest.fixture(scope="session")
def cube_vertices() -> Array:
    cube = TriangleMesh.box(with_top=True)
    triangles_vertices = cube.triangle_vertices

    assert triangles_vertices.shape == (12, 3, 3)

    return triangles_vertices


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
        (jnp.array([0.5, 0.5, 1.0]), jnp.array([0.5, 0.5, -1.0]), jnp.array([True])),
        (jnp.array([0.0, 0.0, 1.0]), jnp.array([1.0, 1.0, -1.0]), jnp.array([True])),
        (jnp.array([0.5, 0.5, 1.0]), jnp.array([0.5, 0.5, +0.5]), jnp.array([False])),
        (jnp.array([0.5, 0.5, 1.0]), jnp.array([1.0, 1.0, +1.0]), jnp.array([False])),
        (jnp.array([0.5, 0.5, 1.0]), jnp.array([1.0, 1.0, +1.5]), jnp.array([False])),
    ],
)
def test_rays_intersect_triangles(
    ray_orig: Array,
    ray_dest: Array,
    expected: Array,
) -> None:
    triangle_vertices = jnp.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    t, hit = rays_intersect_triangles(
        ray_orig,
        ray_dest - ray_orig,
        triangle_vertices,
    )
    got = (t < 1.0) & hit
    chex.assert_trees_all_equal(got, expected)


def test_rays_intersect_triangles_t_and_hit() -> None:
    ray_origin = jnp.array([0.5, 0.5, -1.0])
    ray_directions = jnp.array([
        [0.0, 0.0, +1.0],
        [0.0, 0.0, +0.5],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ])
    triangle_vertices = jnp.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
    ])
    expected_t = jnp.array([[1.0, 2.0], [2.0, 4.0], [-1.0, -2.0], [0.0, 0.0]])
    expected_hit = jnp.array([
        [True, True],
        [True, True],
        [False, False],
        [False, False],
    ])

    got_t, got_hit = rays_intersect_triangles(
        ray_origin[None, None, :],
        ray_directions[:, None, :],
        triangle_vertices,
    )
    chex.assert_trees_all_equal(got_t, expected_t)
    chex.assert_trees_all_equal(got_hit, expected_hit)


@pytest.mark.parametrize(
    ("ray_origins", "ray_directions", "triangle_vertices", "expectation"),
    [
        ((3,), (3,), (3, 3), does_not_raise()),
        ((15, 5, 3), (15, 5, 3), (5, 3, 3), does_not_raise()),
        (
            (15, 5, 3),
            (15, 5, 3),
            (15, 3, 3),
            pytest.raises(TypeError),
        ),
    ],
)
@random_inputs("ray_origins", "ray_directions", "triangle_vertices")
def test_rays_intersect_triangles_random_inputs(
    ray_origins: Array,
    ray_directions: Array,
    triangle_vertices: Array,
    expectation: AbstractContextManager[Exception],
) -> None:
    with expectation:
        got_t, got_hit = rays_intersect_triangles(
            ray_origins,
            ray_directions,
            triangle_vertices,
        )

        assert jnp.where(
            got_hit,
            got_t > 0.0,
            True,  # noqa: FBT003
        ).all(), "t > 0 must be true everywhere hit is true"


@pytest.mark.parametrize(
    ("ray_origins", "ray_directions", "triangle_vertices", "expectation"),
    [
        ((20, 10, 3), (20, 10, 3), (20, 10, 5, 3, 3), does_not_raise()),
        ((10, 3), (10, 3), (1, 3, 3), does_not_raise()),
        ((3,), (3,), (1, 3, 3), does_not_raise()),
        (
            (10, 3),
            (20, 3),
            (1, 3, 3),
            pytest.raises(TypeError),
        ),
        (
            (10, 3),
            (10, 4),
            (10, 3, 3),
            pytest.raises(TypeError),
        ),
    ],
)
@pytest.mark.parametrize("epsilon", [None, 1e-6, 1e-2])
@pytest.mark.parametrize("hit_tol", [None, 0.0, 0.001, -0.5, 0.5])
@random_inputs("ray_origins", "ray_directions", "triangle_vertices")
def test_rays_intersect_any_triangle(
    ray_origins: Array,
    ray_directions: Array,
    triangle_vertices: Array,
    epsilon: float | None,
    hit_tol: float | None,
    expectation: AbstractContextManager[Exception],
) -> None:
    if hit_tol is None:
        dtype = jnp.result_type(ray_origins, ray_directions, triangle_vertices)
        hit_tol = jnp.finfo(dtype).eps

    hit_threshold = 1.0 - hit_tol  # type: ignore[reportOperatorIssue]
    with expectation:
        got = rays_intersect_any_triangle(
            ray_origins,
            ray_directions,
            triangle_vertices,
            epsilon=epsilon,
            hit_tol=hit_tol,
        )
        expected_t, expected_hit = rays_intersect_triangles(
            ray_origins[..., None, :],
            ray_directions[..., None, :],
            triangle_vertices,
            epsilon=epsilon,
        )
        expected = jnp.any((expected_t < hit_threshold) & expected_hit, axis=-1)

        chex.assert_trees_all_equal(got, expected)


@pytest.mark.parametrize(
    ("vertex", "expected_number"),
    [
        (jnp.array([4.0, 0.0, 0.0]), 2),  # Sees one face of the cude
        (jnp.array([4.0, 4.0, 0.0]), 4),  # Sees two faces
        (jnp.array([4.0, 4.0, 4.0]), 6),  # Sees three faces
    ],
)
@pytest.mark.parametrize(
    ("num_rays", "expectation"),
    [
        (
            20,  # Only a few rays are actually needed, thanks to frustum
            does_not_raise(),
        ),
        (10_000, does_not_raise()),
        pytest.param(
            1_000_000,
            does_not_raise(),
            marks=pytest.mark.xfail(
                sys.platform == "win32",
                reason="For some unknown reason, this fails on Windows",
            ),
        ),
        (
            1,  # Impossible to find all visible faces with few rays
            pytest.raises(
                AssertionError,
                match="Number of visible triangles did not match expectation.",
            ),
        ),
    ],
)
def test_triangles_visible_from_vertices(
    vertex: Array,
    expected_number: int,
    num_rays: int,
    expectation: AbstractContextManager[Exception],
    cube_vertices: Array,
) -> None:
    visible_triangles = triangles_visible_from_vertices(
        vertex,
        cube_vertices,
        num_rays=num_rays,
    )

    with expectation:
        assert visible_triangles.sum() == expected_number, (
            "Number of visible triangles did not match expectation."
        )


@pytest.mark.parametrize(
    ("ray_origins", "ray_directions", "triangle_vertices"),
    [
        ((10, 3), (1, 3), (30, 3, 3)),
        ((100, 3), (100, 3), (1, 300, 3, 3)),
    ],
)
@pytest.mark.parametrize("epsilon", [None, 1e-2])
@random_inputs("ray_origins", "ray_directions", "triangle_vertices")
def test_first_triangles_hit_by_rays(
    ray_origins: Array,
    ray_directions: Array,
    triangle_vertices: Array,
    epsilon: float | None,
) -> None:
    got_indices, got_t = first_triangles_hit_by_rays(
        ray_origins,
        ray_directions,
        triangle_vertices,
        epsilon=epsilon,
    )
    expected_t, expected_hit = rays_intersect_triangles(
        ray_origins[..., None, :],
        ray_directions[..., None, :],
        triangle_vertices,
        epsilon=epsilon,
    )
    expected_t = jnp.where(expected_hit, expected_t, jnp.inf)
    expected_indices = jnp.argmin(expected_t, axis=-1)
    assert expected_indices.shape == got_indices.shape
    expected_t = jnp.take_along_axis(
        expected_t, jnp.expand_dims(expected_indices, axis=-1), axis=-1
    ).squeeze(axis=-1)
    assert expected_t.shape == got_t.shape
    expected_indices = jnp.where(expected_t == jnp.inf, -1, expected_indices)

    chex.assert_trees_all_equal(got_indices, expected_indices)
    chex.assert_trees_all_close(got_t, expected_t, rtol=1e-5)
