import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from itertools import product

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, ArrayLike, DTypeLike, Float, PRNGKeyArray

from differt.geometry import Mesh, Scene
from differt.geometry._utils import (
    assemble_path,
    cartesian_to_spherical,
    fibonacci_lattice,
    first_triangle_hit_by_ray,
    generate_all_path_candidates,
    generate_all_path_candidates_chunks_iter,
    generate_all_path_candidates_iter,
    min_distance_between_cells,
    normalize,
    orthogonal_basis,
    path_length,
    perpendicular_vector,
    ray_intersect_any_triangle,
    ray_intersect_triangle,
    rotation_matrix_along_axis,
    rotation_matrix_along_x_axis,
    rotation_matrix_along_y_axis,
    rotation_matrix_along_z_axis,
    spherical_to_cartesian,
    triangles_visible_from_vertex,
    viewing_frustum,
)

from ..utils import random_inputs


@pytest.mark.parametrize(
    "u",
    [
        (10, 3),
        (20, 10, 3),
    ],
)
@pytest.mark.parametrize("keepdims", [False, True])
@random_inputs("u")
def test_normalize(
    u: Array,
    keepdims: bool,
) -> None:
    nu, lu = normalize(u, keepdims=keepdims)

    if keepdims:
        chex.assert_trees_all_close(u, nu * lu)
    else:
        chex.assert_trees_all_close(u, nu * lu[..., None])


def test_perpendicular_vector() -> None:
    all_vectors = list(product([0.0, 1.0, -1.0], repeat=3))
    # Drop [0, 0, 0] case
    all_vectors = all_vectors[1:]
    u = jnp.array(all_vectors)
    v = perpendicular_vector(u)

    chex.assert_trees_all_close(jnp.linalg.norm(v, axis=-1), 1.0)
    chex.assert_trees_all_close(jnp.sum(u * v, axis=-1), 0.0)


@pytest.mark.parametrize(
    "u",
    [
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([0.0, 1.0, 0.0]),
        jnp.array([0.0, 0.0, 1.0]),
        jnp.array([1.0, 1.0, 1.0]),
        jnp.arange(30.0).reshape(2, 5, 3),
    ],
)
def test_orthogonal_basis(u: Array) -> None:
    u, _ = normalize(u)
    v, w = orthogonal_basis(u)

    for vec in [v, w]:
        # Vectors should be perpendicular
        dot = jnp.sum(u * vec, axis=-1)
        chex.assert_trees_all_close(dot, 0.0, atol=1e-7)
        # Vectors should have unit length
        _, length = normalize(vec)
        chex.assert_trees_all_close(length, 1.0)

    # Vectors should be perpendicular
    dot = jnp.sum(u * v, axis=-1)
    chex.assert_trees_all_close(dot, 0.0, atol=1e-7)


@pytest.mark.parametrize(
    "path",
    [
        (10, 3),
        (20, 10, 3),
        (1, 3),
        (0, 3),
    ],
)
@random_inputs("path")
def test_path_length(
    path: Array,
) -> None:
    got = path_length(path)
    expected = jnp.sum(jnp.linalg.norm(jnp.diff(path, axis=-2), axis=-1), axis=-1)

    chex.assert_trees_all_close(got, expected)


@pytest.mark.parametrize("sign", [+1.0, -1.0])
@pytest.mark.parametrize("flip_axis", [False, True])
@pytest.mark.parametrize(
    ("axis", "func"),
    [
        ((1.0, 0.0, 0.0), rotation_matrix_along_x_axis),
        ((0.0, 1.0, 0.0), rotation_matrix_along_y_axis),
        ((0.0, 0.0, 1.0), rotation_matrix_along_z_axis),
    ],
)
def test_rotation_matrices(
    sign: float,
    flip_axis: bool,
    axis: tuple[float, float, float],
    func: Callable[[Float[ArrayLike, ""]], Float[Array, "3 3"]],
    key: PRNGKeyArray,
) -> None:
    angle = jax.random.uniform(key, minval=0.0, maxval=2.0 * jnp.pi)
    angle = jnp.copysign(angle, sign)

    if flip_axis:
        expected = func(-angle)
        got = rotation_matrix_along_axis(angle, -jnp.array(axis))
    else:
        expected = func(+angle)
        got = rotation_matrix_along_axis(angle, +jnp.array(axis))

    chex.assert_trees_all_close(got, expected)


@pytest.mark.parametrize("n", [10, 100])
@pytest.mark.parametrize(
    ("dtype", "expected_dtype", "expectation"),
    [
        (None, jnp.float32, does_not_raise()),
        (jnp.float32, jnp.float32, does_not_raise()),
        (float, jnp.float64, does_not_raise()),
        ("float", jnp.float64, does_not_raise()),
        (jnp.float16, jnp.float16, does_not_raise()),
        (jnp.float64, jnp.float64, does_not_raise()),
        (
            int,
            jnp.float32,
            pytest.raises(
                ValueError,
                match=r"Unsupported dtype <class 'int'>, must be a floating dtype.",
            ),
        ),
        (
            jnp.int32,
            jnp.float32,
            pytest.raises(
                ValueError,
                match=r"Unsupported dtype <class 'jax.numpy.int32'>, must be a floating dtype.",
            ),
        ),
    ],
)
def test_fibonacci_lattice(
    n: int,
    dtype: DTypeLike | None,
    expected_dtype: jnp.dtype,
    expectation: AbstractContextManager[Exception],
) -> None:
    with jax.enable_x64(expected_dtype == jnp.float64), expectation:
        got = fibonacci_lattice(n, dtype=dtype)

        normalized, lengths = normalize(got)

        atol = jnp.finfo(expected_dtype).eps

        chex.assert_type(got, expected_dtype)
        chex.assert_trees_all_close(got, normalized, atol=atol)
        chex.assert_trees_all_close(lengths, jnp.ones_like(lengths), atol=atol)


@pytest.mark.parametrize(
    "n",
    [-1, 0],
)
def test_fibonacci_lattice_neg_n(
    n: int,
) -> None:
    with pytest.raises(
        ValueError, match=f"Invalid size {n!r}, must be strictly positive"
    ):
        _ = fibonacci_lattice(n)


def test_assemble_path() -> None:
    tx = jnp.ones((5, 3))
    paths = jnp.arange(24.0).reshape(4, 2, 3)
    rx = -jnp.ones((6, 3))

    expected = jnp.concatenate(
        (
            jnp.tile(tx[:, None, None, None, :], (1, 6, 4, 1, 1)),
            jnp.tile(paths[None, None, ...], (5, 6, 1, 1, 1)),
            jnp.tile(rx[None, :, None, None, :], (5, 1, 4, 1, 1)),
        ),
        axis=-2,
    )

    got = assemble_path(tx[:, None, None, :], paths, rx[None, :, None, :])

    chex.assert_trees_all_equal(got, expected)


@pytest.mark.parametrize(
    "shapes",
    [
        ((3,), (3,)),
        ((3,), (2, 1, 3), (3,)),
        ((1, 5, 3), (10, 5, 2, 3), (3,)),
    ],
)
def test_assemble_path_various_shapes(
    shapes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    | tuple[tuple[int, ...], tuple[int, ...]],
    key: PRNGKeyArray,
) -> None:
    keys = jax.random.split(key, len(shapes))

    from_vertices = jax.random.uniform(keys[0], shape=shapes[0])
    to_vertices = jax.random.uniform(keys[-1], shape=shapes[-1])
    if len(shapes) == 3:
        intermediate_vertices = jax.random.uniform(keys[1], shape=shapes[1])
    else:
        intermediate_vertices = to_vertices
        to_vertices = None

    _ = assemble_path(from_vertices, intermediate_vertices, to_vertices)


def test_cartesian_to_spherial_roundtrip(key: PRNGKeyArray) -> None:
    key_xyz, key_sph = jax.random.split(key, 2)

    xyz = 10 * jax.random.normal(key_xyz, (100, 3))
    sph = cartesian_to_spherical(xyz)
    got = spherical_to_cartesian(sph)

    chex.assert_trees_all_close(got, xyz, atol=2e-5)

    key_r, key_polar, key_azim = jax.random.split(key_sph, 3)

    r = jnp.abs(10 * jax.random.normal(key_r, (100,)))
    p = jax.random.uniform(key_polar, (100,), minval=0, maxval=jnp.pi)
    a = jax.random.uniform(key_azim, (100,), minval=-jnp.pi, maxval=jnp.pi)
    sph = jnp.stack((r, p, a), axis=-1)
    xyz = spherical_to_cartesian(sph)
    got = cartesian_to_spherical(xyz)

    chex.assert_trees_all_close(got, sph, atol=7e-5)


def test_min_distance_between_cells(key: PRNGKeyArray) -> None:
    key_vertices, key_ids = jax.random.split(key, 2)

    batch = (10, 4, 5)

    cell_vertices = 10 * jax.random.normal(key_vertices, (*batch, 3))
    cell_ids = jax.random.randint(key_ids, batch, minval=0, maxval=5)

    got = min_distance_between_cells(cell_vertices, cell_ids).reshape(-1)

    cell_vertices = cell_vertices.reshape(-1, 3)
    cell_ids = cell_ids.reshape(-1)

    for cell_vertex, cell_id, got_dist in zip(
        cell_vertices, cell_ids, got, strict=False
    ):
        dist = jnp.linalg.norm(cell_vertex - cell_vertices, axis=-1)
        expected_dist = jnp.min(dist, where=(cell_id != cell_ids), initial=jnp.inf)

        chex.assert_trees_all_close(got_dist, expected_dist)


def test_viewing_frustum_narrow_span_no_wraparound() -> None:
    # Vertices in a narrow sector away from ±pi should give a tight frustum.
    # Place viewer at origin, vertices at ~45° azimuth on the XY-plane.
    viewer = jnp.array([0.0, 0.0, 0.0])
    angles = jnp.array([0.6, 0.7, 0.8])  # radians, all in (0, pi)
    # Build 2D vertices at unit distance on the XY-plane (z=0)
    verts = jnp.stack(
        [jnp.cos(angles), jnp.sin(angles), jnp.zeros_like(angles)], axis=-1
    )
    frustum = viewing_frustum(viewer, verts)
    # frustum shape is (2, 3) = [[r_min, p_min, a_min], [r_max, p_max, a_max]]
    a_min, a_max = frustum[0, 2], frustum[1, 2]
    a_width = a_max - a_min
    # Width should be about 0.2 rad, definitely < pi
    assert a_width < jnp.pi, f"Azimuth span {a_width} should be < pi"
    chex.assert_trees_all_close(a_min, 0.6, atol=0.05)
    chex.assert_trees_all_close(a_max, 0.8, atol=0.05)


def test_viewing_frustum_wraparound_at_pi_boundary() -> None:
    # Vertices straddling ±pi should use the [0,2pi) domain for a tighter fit.
    viewer = jnp.array([0.0, 0.0, 0.0])
    # Two vertices near +pi and -pi (actual span ~20°)
    a1 = jnp.pi - 0.15  # ~ +170°
    a2 = -jnp.pi + 0.15  # ~ -170°
    verts = jnp.stack(
        [
            jnp.array([jnp.cos(a1), jnp.sin(a1), 0.0]),
            jnp.array([jnp.cos(a2), jnp.sin(a2), 0.0]),
        ],
        axis=0,
    )
    frustum = viewing_frustum(viewer, verts)
    a_min, a_max = frustum[0, 2], frustum[1, 2]
    a_width = a_max - a_min
    # The naive span would be ~340° (bad); the correct span is ~30° in [0,2pi)
    assert a_width < jnp.pi, (
        f"Azimuth span {a_width:.3f} should be < pi "
        f"(wrap-around should have been handled)"
    )


def test_viewing_frustum_full_circle_fallback_for_surrounding_geometry() -> None:
    # Vertices surrounding the viewer in all azimuthal directions
    # should trigger the full-circle [-pi, pi] fallback.
    viewer = jnp.array([0.0, 0.0, 0.0])
    # Place 8 vertices evenly around the viewer (every 45°)
    angles = jnp.linspace(-jnp.pi, jnp.pi, 8, endpoint=False)
    verts = jnp.stack(
        [jnp.cos(angles), jnp.sin(angles), jnp.zeros_like(angles)], axis=-1
    )
    frustum = viewing_frustum(viewer, verts)
    a_min, a_max = frustum[0, 2], frustum[1, 2]
    # Should fall back to full circle
    chex.assert_trees_all_close(a_min, -jnp.pi, atol=1e-5)
    chex.assert_trees_all_close(a_max, jnp.pi, atol=1e-5)


def test_viewing_frustum_full_circle_fallback_with_corridor_geometry() -> None:
    # Simulate a corridor-like scene (vertices on both sides) where the TX is inside.
    viewer = jnp.array([5.0, 0.0, 1.5])
    # Long corridor: walls on +y and -y side, extending far on both +x and -x.
    wall_y_pos = jnp.array([
        [0.0, 2.0, 0.0],
        [10.0, 2.0, 0.0],
        [0.0, 2.0, 3.0],
        [10.0, 2.0, 3.0],
    ])
    wall_y_neg = jnp.array([
        [0.0, -2.0, 0.0],
        [10.0, -2.0, 0.0],
        [0.0, -2.0, 3.0],
        [10.0, -2.0, 3.0],
    ])
    verts = jnp.concatenate([wall_y_pos, wall_y_neg], axis=0)
    frustum = viewing_frustum(viewer, verts)
    a_min, a_max = frustum[0, 2], frustum[1, 2]
    a_width = a_max - a_min
    # The corridor spans > 270° in azimuth, should trigger full circle
    assert a_width >= 1.9 * jnp.pi, (
        f"Azimuth span {a_width:.3f} should be ~2*pi for corridor geometry"
    )


def test_fibonacci_lattice_output_shape_and_unit_length() -> None:
    # fibonacci_lattice(n) must return n unit vectors on the sphere.
    for n in (1, 10, 1000):
        pts = fibonacci_lattice(n)
        assert pts.shape == (n, 3), f"Expected shape ({n}, 3), got {pts.shape}"
        norms = jnp.linalg.norm(pts, axis=-1)
        chex.assert_trees_all_close(norms, jnp.ones(n), atol=1e-5)


def test_fibonacci_lattice_full_sphere_coverage() -> None:
    # With enough rays, the lattice should cover both hemispheres (z > 0 and z < 0).
    pts = fibonacci_lattice(1000)
    assert (pts[:, 2] > 0).any(), "No points in northern hemisphere"
    assert (pts[:, 2] < 0).any(), "No points in southern hemisphere"


def test_fibonacci_lattice_frustum_constrains_directions() -> None:
    # When a frustum is given, all returned directions must lie within its angular bounds.
    # Build a narrow frustum: polar [pi/4, pi/2], azimuth [0.1, 0.9]
    frustum = jnp.array([
        [0.0, jnp.pi / 4, 0.1],  # [r_min, p_min, a_min]
        [1.0, jnp.pi / 2, 0.9],  # [r_max, p_max, a_max]
    ])
    pts = fibonacci_lattice(500, frustum=frustum)
    assert pts.shape == (500, 3)

    # Convert back to spherical to verify bounds
    sph = cartesian_to_spherical(pts)  # shape (500, 3): [r, polar, azimuth]
    polar = sph[:, 1]
    azimuth = sph[:, 2]

    assert (polar >= jnp.pi / 4 - 1e-4).all(), "Some points below p_min"
    assert (polar <= jnp.pi / 2 + 1e-4).all(), "Some points above p_max"
    assert (azimuth >= 0.1 - 1e-4).all(), "Some points below a_min"
    assert (azimuth <= 0.9 + 1e-4).all(), "Some points above a_max"


def test_fibonacci_lattice_large_n_no_hatching() -> None:
    # At large n the lattice must produce many unique directions (no hatching artifact).
    # The naive float32 computation ``(i * inv_phi) % 1`` loses precision around
    # i ~ 10^7 and collapses to only ~100 distinct azimuthal values, causing visible
    # hatching.  This test confirms both that the naive approach collapses and that
    # the actual function avoids it.
    n = 10_000_000
    inv_phi = jnp.float32(0.6180339887498949)
    i = jnp.arange(n - 10_000, n, dtype=jnp.float32)

    # Naive approach collapses to very few unique fractions (float32 precision loss).
    naive_frac = jnp.array((i * inv_phi) % jnp.float32(1.0))
    assert len(jnp.unique(naive_frac)) < 1000

    # The actual function must still produce many distinct directions in the tail.
    pts = fibonacci_lattice(n)
    tail = jnp.array(jnp.round(pts[-10_000:], decimals=4))
    assert len(jnp.unique(tail, axis=0)) > 5000


@pytest.fixture(scope="session")
def cube_vertices() -> Array:
    cube = Mesh.box(with_top=True)
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
def test_ray_intersect_triangle(
    ray_orig: Array,
    ray_dest: Array,
    expected: Array,
) -> None:
    triangle_vertices = jnp.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    t, hit = ray_intersect_triangle(
        ray_orig,
        ray_dest - ray_orig,
        triangle_vertices,
    )
    got = (t < 1.0) & hit
    chex.assert_trees_all_equal(got, expected)


def test_ray_intersect_triangle_t_and_hit() -> None:
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

    got_t, got_hit = ray_intersect_triangle(
        ray_origin[None, None, :],
        ray_directions[:, None, :],
        triangle_vertices,
    )
    chex.assert_trees_all_equal(got_t, expected_t)
    chex.assert_trees_all_equal(got_hit, expected_hit)


@pytest.mark.parametrize(
    ("ray_origins", "ray_directions", "triangle_vertices"),
    [
        ((3,), (3,), (3, 3)),
        ((15, 5, 3), (15, 5, 3), (5, 3, 3)),
    ],
)
@random_inputs("ray_origins", "ray_directions", "triangle_vertices")
def test_ray_intersect_triangle_various(
    ray_origins: Array,
    ray_directions: Array,
    triangle_vertices: Array,
) -> None:
    got_t, got_hit = ray_intersect_triangle(
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

    got_t, got_hit = ray_intersect_triangle(
        ray_origins,
        ray_directions,
        triangle_vertices,
        smoothing_factor=1e8,
    )

    chex.assert_trees_all_equal(got_t, expected_t)
    chex.assert_trees_all_equal(got_hit > 0.5, expected_hit)


@pytest.mark.parametrize(
    ("ray_origins", "ray_directions", "triangle_vertices"),
    [
        ((20, 10, 3), (20, 10, 3), (20, 10, 5, 3, 3)),
        ((10, 3), (10, 3), (1, 3, 3)),
        ((3,), (3,), (1, 3, 3)),
    ],
)
@pytest.mark.parametrize("epsilon", [None, 1e-6, 1e-2])
@pytest.mark.parametrize("hit_tol", [None, 0.0, 0.001, -0.5, 0.5])
@pytest.mark.parametrize("with_active_triangles", [True, False])
@random_inputs("ray_origins", "ray_directions", "triangle_vertices")
def test_ray_intersect_any_triangle(
    ray_origins: Array,
    ray_directions: Array,
    triangle_vertices: Array,
    epsilon: float | None,
    hit_tol: float | None,
    with_active_triangles: bool,
) -> None:
    if hit_tol is None:
        dtype = jnp.result_type(ray_origins, ray_directions, triangle_vertices)
        hit_tol_arr = 100.0 * jnp.finfo(dtype).eps
    else:
        hit_tol_arr = jnp.asarray(hit_tol)

    active_triangles = (
        jnp.ones(triangle_vertices.shape[:-2], dtype=bool)
        if with_active_triangles
        else None
    )

    hit_threshold = 1.0 - hit_tol_arr
    got = ray_intersect_any_triangle(
        ray_origins,
        ray_directions,
        triangle_vertices,
        active_triangles=active_triangles,
        epsilon=epsilon,
        hit_tol=hit_tol,
        batch_size=11,  # will create non-zero remainder
    )
    expected_t, expected_hit = ray_intersect_triangle(
        ray_origins[..., None, :],
        ray_directions[..., None, :],
        triangle_vertices,
        epsilon=epsilon,
    )
    expected = jnp.any((expected_t < hit_threshold) & expected_hit, axis=-1)

    chex.assert_trees_all_equal(got, expected)

    # Check that large smoothing factor matches no smoothing
    # TODO: fixme

    got = ray_intersect_any_triangle(
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
def test_triangles_visible_from_vertex(
    vertex: Array,
    expected_number: int,
    num_rays: int,
    expectation: AbstractContextManager[Exception],
    cube_vertices: Array,
) -> None:
    visible_triangles = triangles_visible_from_vertex(
        vertex,
        cube_vertices,
        num_rays=num_rays,
        batch_size=1,
    )

    with expectation:
        assert visible_triangles.sum() == expected_number, (
            "Number of visible triangles did not match expectation."
        )


def test_triangles_visible_from_vertex_inside_box() -> None:
    outer_mesh = Mesh.box(4.0, 4.0, 4.0)
    inner_mesh = Mesh.box(1.0, 1.0, 1.0)
    mesh = outer_mesh + inner_mesh

    # Mask to keep only the outer mesh
    mask = jnp.concatenate((
        jnp.ones((outer_mesh.num_triangles,), dtype=bool),
        jnp.zeros((inner_mesh.num_triangles,), dtype=bool),
    ))
    mesh = eqx.tree_at(lambda m: m.mask, mesh, mask, is_leaf=lambda x: x is None)

    tx = jnp.array([-1.0, 0.0, 0.0])
    rx = jnp.array([+1.0, 0.0, 0.0])

    visible_triangles_from_tx = triangles_visible_from_vertex(
        tx,
        mesh.triangle_vertices,
        mesh.mask,
    )

    visible_triangles_from_rx = triangles_visible_from_vertex(
        rx,
        mesh.triangle_vertices,
        mesh.mask,
    )

    chex.assert_trees_all_equal(visible_triangles_from_tx, visible_triangles_from_rx)
    assert visible_triangles_from_tx.sum() == 10, (
        "Should see all 10 triangles from either side"
    )
    chex.assert_trees_all_equal(visible_triangles_from_tx, mask)

    visible_triangles_from_tx_ignore_mask = triangles_visible_from_vertex(
        tx,
        mesh.triangle_vertices,
        None,
    )

    visible_triangles_from_rx_ignore_mask = triangles_visible_from_vertex(
        rx,
        mesh.triangle_vertices,
        None,
    )

    assert (
        visible_triangles_from_tx_ignore_mask != visible_triangles_from_rx_ignore_mask
    ).any(), (
        "TX and RX should see different triangles of the inner mesh when not using the mask"
    )

    # N.B.: the viewing frustum aligns with the inner box so it perfectly hit the edge of one additional triangle for TX and two for RX
    assert visible_triangles_from_tx_ignore_mask.sum() == 11, (
        "Should see 11 triangles from TX when ignoring mask"
    )
    assert visible_triangles_from_rx_ignore_mask.sum() == 12, (
        "Should see 12 triangles from RX when ignoring mask"
    )


def test_triangles_visible_from_vertex_street_canyon(
    simple_street_canyon_scene: Scene,
) -> None:
    scene = simple_street_canyon_scene
    tx = jnp.array([-35, 0, 32.0])
    rx = jnp.array([+35, 0, 1.5])
    visible_triangles = triangles_visible_from_vertex(
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

    visible_triangles = triangles_visible_from_vertex(
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
def test_first_triangle_hit_by_ray(
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
    got_indices, got_t = first_triangle_hit_by_ray(
        ray_origins,
        ray_directions,
        triangle_vertices,
        active_triangles=active_triangles,
        epsilon=epsilon,
        batch_size=11,  # will create non-zero remainder
    )
    expected_t, expected_hit = ray_intersect_triangle(
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
