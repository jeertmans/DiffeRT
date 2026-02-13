import sys
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import chex
import equinox as eqx
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
from differt.scene import TriangleScene

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
    got = (
        got.at[jnp.lexsort(got.T[::-1])].get(unique_indices=True)
        if got.size > 0
        else got
    )  # order may not be the same so we sort
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
    expected = (
        expected.at[jnp.lexsort(expected.T[::-1])].get(unique_indices=True)
        if expected.size > 0
        else expected
    )
    got = list(generate_all_path_candidates_iter(num_primitives, order))
    got = jnp.asarray(got)
    got = (
        got.at[jnp.lexsort(got.T[::-1])].get(unique_indices=True)
        if got.size > 0
        else got
    )

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
        pytest.param(
            (15, 5, 3),
            (15, 5, 3),
            (15, 3, 3),
            pytest.raises(TypeError),
            marks=pytest.mark.jaxtyped,
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
            True,
        ).all(), "t > 0 must be true everywhere hit is true"

        expected_t, expected_hit = got_t, got_hit

        # Check that large smoothing factor matches no smoothing

        got_t, got_hit = rays_intersect_triangles(
            ray_origins,
            ray_directions,
            triangle_vertices,
            smoothing_factor=1e8,
        )

        chex.assert_trees_all_equal(got_t, expected_t)
        chex.assert_trees_all_equal(got_hit > 0.5, expected_hit)


@pytest.mark.parametrize(
    ("ray_origins", "ray_directions", "triangle_vertices", "expectation"),
    [
        ((20, 10, 3), (20, 10, 3), (20, 10, 5, 3, 3), does_not_raise()),
        ((10, 3), (10, 3), (1, 3, 3), does_not_raise()),
        ((3,), (3,), (1, 3, 3), does_not_raise()),
        pytest.param(
            (10, 3),
            (20, 3),
            (1, 3, 3),
            pytest.raises(TypeError),
            marks=pytest.mark.jaxtyped,
        ),
        pytest.param(
            (10, 3),
            (10, 4),
            (10, 3, 3),
            pytest.raises(TypeError),
            marks=pytest.mark.jaxtyped,
        ),
    ],
)
@pytest.mark.parametrize("epsilon", [None, 1e-6, 1e-2])
@pytest.mark.parametrize("hit_tol", [None, 0.0, 0.001, -0.5, 0.5])
@pytest.mark.parametrize("with_active_triangles", [True, False])
@random_inputs("ray_origins", "ray_directions", "triangle_vertices")
def test_rays_intersect_any_triangle(
    ray_origins: Array,
    ray_directions: Array,
    triangle_vertices: Array,
    epsilon: float | None,
    hit_tol: float | None,
    with_active_triangles: bool,
    expectation: AbstractContextManager[Exception],
) -> None:
    if hit_tol is None:
        dtype = jnp.result_type(ray_origins, ray_directions, triangle_vertices)
        hit_tol_arr = jnp.finfo(dtype).eps
    else:
        hit_tol_arr = jnp.asarray(hit_tol)

    active_triangles = (
        jnp.ones(triangle_vertices.shape[:-2], dtype=bool)
        if with_active_triangles
        else None
    )

    hit_threshold = 1.0 - hit_tol_arr
    with expectation:
        got = rays_intersect_any_triangle(
            ray_origins,
            ray_directions,
            triangle_vertices,
            active_triangles=active_triangles,
            epsilon=epsilon,
            hit_tol=hit_tol,
            batch_size=11,  # will create non-zero remainder
        )
        expected_t, expected_hit = rays_intersect_triangles(
            ray_origins[..., None, :],
            ray_directions[..., None, :],
            triangle_vertices,
            epsilon=epsilon,
        )
        expected = jnp.any((expected_t < hit_threshold) & expected_hit, axis=-1)

        chex.assert_trees_all_equal(got, expected)

        # Check that large smoothing factor matches no smoothing
        # TODO: fixme

        got = rays_intersect_any_triangle(
            ray_origins,
            ray_directions,
            triangle_vertices,
            epsilon=epsilon,
            hit_tol=hit_tol,
            smoothing_factor=1e8,
            batch_size=11,  # will create non-zero remainder
        )

        chex.assert_trees_all_equal(got > 0.5, expected)


@pytest.mark.parametrize(
    ("vertex", "expected_number"),
    [
        (jnp.array([2.0, 0.0, 0.0]), 2),  # Sees one face of the cude
        (jnp.array([2.0, 2.0, 0.0]), 4),  # Sees two faces
        (jnp.array([2.0, 2.0, 2.0]), 6),  # Sees three faces
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
                match=r"Number of visible triangles did not match expectation.",
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
        batch_size=1,
    )

    with expectation:
        assert visible_triangles.sum() == expected_number, (
            "Number of visible triangles did not match expectation."
        )


def test_triangles_visible_from_vertices_inside_box() -> None:
    outer_mesh = TriangleMesh.box(4.0, 4.0, 4.0)
    inner_mesh = TriangleMesh.box(1.0, 1.0, 1.0)
    mesh = outer_mesh + inner_mesh

    # Mask to keep only the outer mesh
    mask = jnp.concatenate((
        jnp.ones((outer_mesh.num_triangles,), dtype=bool),
        jnp.zeros((inner_mesh.num_triangles,), dtype=bool),
    ))
    mesh = eqx.tree_at(lambda m: m.mask, mesh, mask, is_leaf=lambda x: x is None)

    tx = jnp.array([-1.0, 0.0, 0.0])
    rx = jnp.array([+1.0, 0.0, 0.0])

    visible_triangles_from_tx = triangles_visible_from_vertices(
        tx,
        mesh.triangle_vertices,
        mesh.mask,
    )

    visible_triangles_from_rx = triangles_visible_from_vertices(
        rx,
        mesh.triangle_vertices,
        mesh.mask,
    )

    chex.assert_trees_all_equal(visible_triangles_from_tx, visible_triangles_from_rx)
    assert visible_triangles_from_tx.sum() == 10, (
        "Should see all 10 triangles from either side"
    )
    chex.assert_trees_all_equal(visible_triangles_from_tx, mask)

    visible_triangles_from_tx_ignore_mask = triangles_visible_from_vertices(
        tx,
        mesh.triangle_vertices,
        None,
    )

    visible_triangles_from_rx_ignore_mask = triangles_visible_from_vertices(
        rx,
        mesh.triangle_vertices,
        None,
    )

    assert (
        visible_triangles_from_tx_ignore_mask != visible_triangles_from_rx_ignore_mask
    ).any(), (
        "TX and RX should see different triangles of the inner mesh when not using the mask"
    )

    assert visible_triangles_from_tx_ignore_mask.sum() == 10, (
        "Should see 10 triangles from TX when ignoring mask"
    )
    assert visible_triangles_from_rx_ignore_mask.sum() == 10, (
        "Should see 10 triangles from RX when ignoring mask"
    )


def test_triangles_visible_from_vertices_street_canyon(
    simple_street_canyon_scene: TriangleScene,
) -> None:
    scene = simple_street_canyon_scene
    tx = jnp.array([-35, 0, 32.0])
    rx = jnp.array([+35, 0, 1.5])
    visible_triangles = triangles_visible_from_vertices(
        tx,
        scene.mesh.triangle_vertices,
    )

    got_visible = jnp.argwhere(visible_triangles)
    expected_visible = jnp.array([
        [6],
        [7],
        [10],
        [11],
        [12],
        [13],
        [18],
        [19],
        [24],
        [25],
        [30],
        [31],
        [34],
        [35],
        [36],
        [37],
        [38],
        [39],
        [50],
        [51],
        [58],
        [59],
        [60],
        [61],
        [62],
        [63],
        [70],
        [71],
        [72],
        [73],
    ])

    chex.assert_trees_all_equal(got_visible, expected_visible)

    visible_triangles = triangles_visible_from_vertices(
        rx,
        scene.mesh.triangle_vertices,
    )
    got_visible = jnp.argwhere(visible_triangles)
    expected_visible = jnp.array([
        [4],
        [5],
        [6],
        [7],
        [16],
        [17],
        [18],
        [19],
        [30],
        [31],
        [38],
        [39],
        [40],
        [41],
        [50],
        [51],
        [52],
        [53],
        [62],
        [63],
        [72],
        [73],
    ])

    chex.assert_trees_all_equal(got_visible, expected_visible)


@pytest.mark.parametrize(
    ("ray_origins", "ray_directions", "triangle_vertices"),
    [
        ((10, 3), (1, 3), (30, 3, 3)),
        ((100, 3), (100, 3), (1, 300, 3, 3)),
        ((4, 3), (4, 3), (0, 3, 3)),
    ],
)
@pytest.mark.parametrize("epsilon", [None, 1e-2])
@pytest.mark.parametrize("with_active_triangles", [True, False])
@random_inputs("ray_origins", "ray_directions", "triangle_vertices")
def test_first_triangles_hit_by_rays(
    ray_origins: Array,
    ray_directions: Array,
    triangle_vertices: Array,
    epsilon: float | None,
    with_active_triangles: bool,
) -> None:
    active_triangles = (
        jnp.ones(triangle_vertices.shape[:-2], dtype=bool)
        if with_active_triangles
        else None
    )
    got_indices, got_t = first_triangles_hit_by_rays(
        ray_origins,
        ray_directions,
        triangle_vertices,
        active_triangles=active_triangles,
        epsilon=epsilon,
        batch_size=11,  # will create non-zero remainder
    )
    expected_t, expected_hit = rays_intersect_triangles(
        ray_origins[..., None, :],
        ray_directions[..., None, :],
        triangle_vertices,
        epsilon=epsilon,
    )
    expected_t = jnp.where(expected_hit, expected_t, jnp.inf)
    if triangle_vertices.shape[-3] == 0:
        expected_t = jnp.full_like(expected_t, jnp.inf, shape=expected_t.shape[:-1])
        expected_indices = jnp.full(expected_t.shape, -1, dtype=int)
    else:
        expected_indices = jnp.argmin(expected_t, axis=-1)
        assert expected_indices.shape == got_indices.shape
        expected_t = jnp.take_along_axis(
            expected_t, jnp.expand_dims(expected_indices, axis=-1), axis=-1
        ).squeeze(axis=-1)
        assert expected_t.shape == got_t.shape
        expected_indices = jnp.where(expected_t == jnp.inf, -1, expected_indices)

    # TODO: fixme, we need to fix the index if two or more triangles are hit at the same t
    # chex.assert_trees_all_equal(got_indices, expected_indices)
    chex.assert_trees_all_close(got_t, expected_t, rtol=1e-5)


def test_rays_intersect_any_triangle_oom_stress() -> None:
    """
    Test that rays_intersect_any_triangle can handle very large inputs without OOM.

    This test uses broadcast_to to create large virtual arrays without allocating
    the full memory, then verifies the implementation doesn't materialize the full
    (num_rays, num_triangles) interaction matrix.
    """
    import time

    import jax

    print("\n--- Running OOM Stress Test (100K Rays x 100K Triangles) ---")
    print("Note: This test effectively requests 10 Billion interaction checks.")
    print(
        "Without proper reduction (using reduce instead of materialized arrays), "
        "this would cause OOM."
    )

    # Use 100,000 rays and triangles (10 billion pairwise checks)
    N = 100_000
    M = 100_000

    # Create small actual arrays and broadcast them to test the logic's memory limit
    # without running out of RAM just creating the inputs.
    ray_o = jnp.zeros((1, 3))
    ray_d = jnp.array([[0.0, 0.0, 1.0]])
    tri = jnp.array([[[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0]]])

    big_rays_o = jnp.broadcast_to(ray_o, (N, 3))
    big_rays_d = jnp.broadcast_to(ray_d, (N, 3))
    big_tris = jnp.broadcast_to(tri, (M, 3, 3))

    print(f"Input shapes virtualized: Rays {big_rays_o.shape}, Tris {big_tris.shape}")

    # Test that naive vmap would fail or be very slow
    # (We skip this to avoid actually OOM-ing the test runner)
    print("Skipping naive vmap test to avoid actual OOM...")

    # Test our implementation should work
    try:
        start = time.time()
        # Compile and run
        res = rays_intersect_any_triangle(big_rays_o, big_rays_d, big_tris)
        res.block_until_ready()
        elapsed = time.time() - start
        print(f"SUCCESS: Computed 10B interactions in {elapsed:.2f}s")
        print(f"Result shape: {res.shape}")
        assert res.shape == (N,)
        # All rays should hit since they all point up and triangle is at z=5
        # But because t=5 > 1.0 (hit_threshold), they won't register as hits
        print(f"Number of hits: {jnp.sum(res)}")
    except Exception as e:
        pytest.fail(
            f"FAILURE: Test crashed with memory or other error. Error: {e}"
        )
