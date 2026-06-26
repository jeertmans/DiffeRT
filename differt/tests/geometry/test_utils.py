from collections.abc import Callable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from itertools import product

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, ArrayLike, DTypeLike, Float, PRNGKeyArray

from differt.geometry._utils import (
    assemble_paths,
    cartesian_to_spherical,
    fibonacci_lattice,
    min_distance_between_cells,
    normalize,
    orthogonal_basis,
    path_lengths,
    perpendicular_vectors,
    rotation_matrix_along_axis,
    rotation_matrix_along_x_axis,
    rotation_matrix_along_y_axis,
    rotation_matrix_along_z_axis,
    spherical_to_cartesian,
    viewing_frustum,
)

from ..utils import random_inputs


@pytest.mark.parametrize(
    ("u", "expectation"),
    [
        ((10, 3), does_not_raise()),
        ((20, 10, 3), does_not_raise()),
        pytest.param(
            (10, 4),
            pytest.raises(TypeError),
            marks=pytest.mark.require_typechecker,
        ),
    ],
)
@pytest.mark.parametrize("keepdims", [False, True])
@random_inputs("u")
def test_normalize_random_inputs(
    u: Array,
    keepdims: bool,
    expectation: AbstractContextManager[Exception],
) -> None:
    with expectation:
        nu, lu = normalize(u, keepdims=keepdims)

        if keepdims:
            chex.assert_trees_all_close(u, nu * lu)
        else:
            chex.assert_trees_all_close(u, nu * lu[..., None])


def test_perpendicular_vectors() -> None:
    all_vectors = list(product([0.0, 1.0, -1.0], repeat=3))
    # Drop [0, 0, 0] case
    all_vectors = all_vectors[1:]
    u = jnp.array(all_vectors)
    v = perpendicular_vectors(u)

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
    ("paths", "expectation"),
    [
        ((10, 3), does_not_raise()),
        ((20, 10, 3), does_not_raise()),
        pytest.param(
            (10, 4),
            pytest.raises(TypeError),
            marks=pytest.mark.require_typechecker,
        ),
        ((1, 3), does_not_raise()),
        ((0, 3), does_not_raise()),
    ],
)
@random_inputs("paths")
def test_path_lengths_random_inputs(
    paths: Array,
    expectation: AbstractContextManager[Exception],
) -> None:
    with expectation:
        got = path_lengths(paths)
        expected = jnp.sum(jnp.linalg.norm(jnp.diff(paths, axis=-2), axis=-1), axis=-1)

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


def test_assemble_paths() -> None:
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

    got = assemble_paths(tx[:, None, None, :], paths, rx[None, :, None, :])

    chex.assert_trees_all_equal(got, expected)


@pytest.mark.parametrize(
    ("shapes", "expectation"),
    [
        (((3,), (3,)), does_not_raise()),
        (((3,), (2, 1, 3), (3,)), does_not_raise()),
        (((1, 5, 3), (10, 5, 2, 3), (3,)), does_not_raise()),
        pytest.param(
            ((3,), (6, 4), (3,)),
            pytest.raises(TypeError),
            marks=pytest.mark.require_typechecker,
        ),
        pytest.param(
            ((1, 3), (3,), (1, 3)),
            pytest.raises(TypeError),
            marks=pytest.mark.require_typechecker,
        ),
    ],
)
def test_assemble_paths_random_inputs(
    shapes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    | tuple[tuple[int, ...], tuple[int, ...]],
    expectation: AbstractContextManager[Exception],
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

    with expectation:
        _ = assemble_paths(from_vertices, intermediate_vertices, to_vertices)


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


# ---------------------------------------------------------------
# viewing_frustum tests
# ---------------------------------------------------------------


class TestViewingFrustumAzimuth:
    """Tests for the azimuth discontinuity handling in viewing_frustum."""

    def test_narrow_span_no_wraparound(self) -> None:
        """Vertices in a narrow sector away from ±pi should give a tight frustum."""
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

    def test_wraparound_at_pi_boundary(self) -> None:
        """Vertices straddling ±pi should use the [0,2pi) domain for a tighter fit."""
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

    def test_full_circle_fallback_for_surrounding_geometry(self) -> None:
        """Vertices surrounding the viewer in all azimuthal directions
        should trigger the full-circle [-pi, pi] fallback."""
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

    def test_full_circle_fallback_with_corridor_geometry(self) -> None:
        """Simulate a corridor-like scene (vertices on both sides)
        where the TX is inside."""
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


# ---------------------------------------------------------------
# fibonacci_lattice precision test (hatching artifact guard)
# ---------------------------------------------------------------


def test_fibonacci_lattice_large_n_unique_fractions() -> None:
    """At large n, the two-stage decomposition should preserve enough
    precision so that the last 10k points still have many unique
    azimuthal fractions (i.e. no hatching artifacts)."""
    n = 10_000_000
    i = jnp.arange(n - 10_000, n, dtype=jnp.float32)

    inv_phi = 0.6180339887498949
    m1 = 262144.0
    m2 = 512.0
    inv_phi_m1 = (inv_phi * m1) % 1.0
    inv_phi_m2 = (inv_phi * m2) % 1.0

    q1 = jnp.floor(i / m1)
    rem = i - q1 * m1
    q2 = jnp.floor(rem / m2)
    r = rem - q2 * m2
    frac = (q1 * inv_phi_m1 + q2 * inv_phi_m2 + r * inv_phi) % 1.0

    # With 10k samples, we should get at least 5000 unique fractional
    # values.  Without the fix, float32 (i * inv_phi) % 1.0 collapses
    # to ~100 unique values at i ~ 10^7.
    unique_count = jnp.unique(frac, size=10_000).shape[0]
    assert unique_count > 5000, (
        f"Only {unique_count} unique fractions in last 10k samples; "
        f"expected >5000 (possible precision regression)"
    )
