# ruff: file-ignore[non-lowercase-variable-in-function]
"""Integration tests validating DiffeRT against theoretical equations in McNamara's UTD book.

References:
    D. A. McNamara, C. W. I. Pistorius, and J. A. G. Malherbe,
    "Introduction to the Uniform Geometrical Theory of Diffraction",
    Artech House, 1990.
"""

from typing import Literal

import chex
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Complex, Float

from differt.em import (
    F,
    GeometricFieldSolver,
    InteractionType,
    L_i,
    WavefrontState,
    diffraction_coefficients,
    reflection_coefficients,
    sp_directions,
    z_0,
)
from differt.geometry import Mesh, Scene, normalize


@pytest.fixture
def flat_plate_scene() -> Scene:
    """A flat PEC plate at z = 0 with a transmitter and receiver."""
    vertices = jnp.array([
        [-5.0, -5.0, 0.0],
        [5.0, -5.0, 0.0],
        [5.0, 5.0, 0.0],
        [-5.0, 5.0, 0.0],
    ])
    triangles = jnp.array([
        [0, 1, 2],
        [0, 2, 3],
    ])
    mesh = Mesh(
        vertices=vertices,
        triangles=triangles,
        face_materials=jnp.array([0, 0]),
        material_names=("Metal",),
    )
    return Scene(
        transmitters=jnp.array([0.0, 0.0, 2.0]),
        receivers=jnp.array([2.0, 0.0, 2.0]),
        mesh=mesh,
    )


@pytest.fixture
def wedge_scene() -> Scene:
    """A right-angle convex wedge with one diffraction edge along y."""
    vertices = jnp.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [1.0, 0.0, -1.0],  # 3
    ])
    triangles = jnp.array([
        [0, 1, 2],
        [1, 3, 2],
    ])
    mesh = Mesh(
        vertices=vertices,
        triangles=triangles,
        assume_quads=False,
        face_materials=jnp.array([0, 0]),
        material_names=("Metal",),
    )
    return Scene(
        transmitters=jnp.array([0.5, 0.5, 1.0]),
        receivers=jnp.array([2.0, 0.5, -0.5]),
        mesh=mesh,
    )


def test_eq_3_1_incident_go_ray_tube_astigmatic_field() -> None:
    """Eq. (3.1), p. 66: Field variation along an incident astigmatic GO ray tube."""
    rho_1_i = 5.0
    rho_2_i = 8.0
    s_i = 10.0

    amplitude_ratio_expected = jnp.sqrt(
        (rho_1_i * rho_2_i) / ((rho_1_i + s_i) * (rho_2_i + s_i))
    )

    k_hat = jnp.array([0.0, 0.0, 1.0])
    wf = WavefrontState.from_tx(k_hat, (rho_1_i, rho_2_i)).propagate(jnp.asarray(s_i))
    chex.assert_trees_all_close(wf.radii, jnp.array([rho_1_i + s_i, rho_2_i + s_i]))

    spreading_init = 1.0 / jnp.sqrt(rho_1_i * rho_2_i)
    spreading_propagated = 1.0 / jnp.sqrt((rho_1_i + s_i) * (rho_2_i + s_i))
    chex.assert_trees_all_close(
        spreading_propagated / spreading_init, amplitude_ratio_expected
    )


def test_eq_3_2_reflected_go_ray_tube_field(flat_plate_scene: Scene) -> None:
    """Eq. (3.2), p. 68: Reflected GO field at observation point P from specular point Q_r."""
    paths = flat_plate_scene.trace_paths(order=1, solver="exhaustive").masked()
    assert paths.num_valid_paths == 1
    v = paths.vertices[paths.mask][0]
    s_i = jnp.linalg.norm(v[1] - v[0])
    s_r = jnp.linalg.norm(v[2] - v[1])

    rho_1_r = s_i
    rho_2_r = s_i
    amplitude_factor_expected = jnp.sqrt(
        (rho_1_r * rho_2_r) / ((rho_1_r + s_r) * (rho_2_r + s_r))
    )
    chex.assert_trees_all_close(amplitude_factor_expected, s_i / (s_i + s_r))

    solver = GeometricFieldSolver()
    spreading = solver.spreading_factor(paths, flat_plate_scene.mesh)[paths.mask][0]
    chex.assert_trees_all_close(spreading, 1.0 / (s_i + s_r))


def test_eqs_3_3_to_3_6_ray_fixed_polarization_basis_vectors(
    flat_plate_scene: Scene,
) -> None:
    """Eqs. (3.3)-(3.6), p. 74: Ray-fixed basis vectors for incident and reflected rays."""
    paths = flat_plate_scene.trace_paths(order=1, solver="exhaustive").masked()
    v = paths.vertices[paths.mask][0]
    s_i_hat = (v[1] - v[0]) / jnp.linalg.norm(v[1] - v[0])
    s_r_hat = (v[2] - v[1]) / jnp.linalg.norm(v[2] - v[1])
    n = jnp.array([0.0, 0.0, 1.0])

    e_par_i_expected = normalize(jnp.cross(s_i_hat, jnp.cross(n, s_i_hat)))[0]
    e_par_r_expected = normalize(jnp.cross(s_r_hat, jnp.cross(n, s_r_hat)))[0]

    (e_i_s, e_i_p), (e_r_s, e_r_p) = sp_directions(
        s_i_hat[None, :], s_r_hat[None, :], n[None, :]
    )
    e_perp_i, e_par_i = e_i_s[0], e_i_p[0]
    e_perp_r, e_par_r = e_r_s[0], e_r_p[0]

    chex.assert_trees_all_close(jnp.cross(e_perp_i, s_i_hat), e_par_i, atol=1e-6)
    chex.assert_trees_all_close(jnp.cross(e_perp_r, s_r_hat), e_par_r, atol=1e-6)
    chex.assert_trees_all_close(e_par_i, e_par_i_expected, atol=1e-6)
    chex.assert_trees_all_close(e_par_r, e_par_r_expected, atol=1e-6)


def test_eqs_3_7_3_8_surface_normal_cross_parallel_polarization(
    flat_plate_scene: Scene,
) -> None:
    """Eqs. (3.7)-(3.8), p. 74: Surface normal cross parallel polarization vector."""
    paths = flat_plate_scene.trace_paths(order=1, solver="exhaustive").masked()
    v = paths.vertices[paths.mask][0]
    s_i_hat = (v[1] - v[0]) / jnp.linalg.norm(v[1] - v[0])
    s_r_hat = (v[2] - v[1]) / jnp.linalg.norm(v[2] - v[1])
    n = jnp.array([0.0, 0.0, 1.0])

    (e_i_s, e_i_p), (e_r_s, e_r_p) = sp_directions(
        s_i_hat[None, :], s_r_hat[None, :], n[None, :]
    )
    e_perp_i, e_par_i = e_i_s[0], e_i_p[0]
    e_perp_r, e_par_r = e_r_s[0], e_r_p[0]

    chex.assert_trees_all_close(
        jnp.cross(n, e_par_i), jnp.dot(n, s_i_hat) * e_perp_i, atol=1e-6
    )
    chex.assert_trees_all_close(
        jnp.cross(n, e_par_r), jnp.dot(n, s_r_hat) * e_perp_r, atol=1e-6
    )


def test_eq_3_17_law_of_reflection_normal_component(
    flat_plate_scene: Scene,
) -> None:
    """Eq. (3.17), p. 77: Normal components of incident and reflected rays are opposite."""
    paths = flat_plate_scene.trace_paths(order=1, solver="exhaustive").masked()
    v = paths.vertices[paths.mask][0]
    s_i_hat = (v[1] - v[0]) / jnp.linalg.norm(v[1] - v[0])
    s_r_hat = (v[2] - v[1]) / jnp.linalg.norm(v[2] - v[1])
    n = jnp.array([0.0, 0.0, 1.0])

    chex.assert_trees_all_close(jnp.dot(n, s_i_hat), -jnp.dot(n, s_r_hat), atol=1e-6)


def test_eq_3_18_equality_of_incidence_and_reflection_angles(
    flat_plate_scene: Scene,
) -> None:
    """Eq. (3.18), p. 77: Angle of incidence equals angle of reflection (theta^i = theta^r)."""
    paths = flat_plate_scene.trace_paths(order=1, solver="exhaustive").masked()
    v = paths.vertices[paths.mask][0]
    s_i_hat = (v[1] - v[0]) / jnp.linalg.norm(v[1] - v[0])
    s_r_hat = (v[2] - v[1]) / jnp.linalg.norm(v[2] - v[1])
    n = jnp.array([0.0, 0.0, 1.0])

    cos_theta_i = -jnp.dot(n, s_i_hat)
    cos_theta_r = jnp.dot(n, s_r_hat)
    theta_i = jnp.arccos(jnp.clip(cos_theta_i, -1.0, 1.0))
    theta_r = jnp.arccos(jnp.clip(cos_theta_r, -1.0, 1.0))

    chex.assert_trees_all_close(theta_i, theta_r, atol=1e-6)


def test_eq_3_19_law_of_reflection_coplanarity(
    flat_plate_scene: Scene,
) -> None:
    """Eq. (3.19), p. 77: Coplanarity of surface normal and incident/reflected rays."""
    paths = flat_plate_scene.trace_paths(order=1, solver="exhaustive").masked()
    v = paths.vertices[paths.mask][0]
    s_i_hat = (v[1] - v[0]) / jnp.linalg.norm(v[1] - v[0])
    s_r_hat = (v[2] - v[1]) / jnp.linalg.norm(v[2] - v[1])
    n = jnp.array([0.0, 0.0, 1.0])

    chex.assert_trees_all_close(jnp.cross(n, s_i_hat), jnp.cross(n, s_r_hat), atol=1e-6)


def test_eq_3_23_vector_law_of_reflection(flat_plate_scene: Scene) -> None:
    """Eq. (3.23), p. 79: Vector law of reflection s^r = s^i - 2(n . s^i)n."""
    paths = flat_plate_scene.trace_paths(order=1, solver="exhaustive").masked()
    v = paths.vertices[paths.mask][0]
    s_i_hat = (v[1] - v[0]) / jnp.linalg.norm(v[1] - v[0])
    s_r_hat = (v[2] - v[1]) / jnp.linalg.norm(v[2] - v[1])
    n = jnp.array([0.0, 0.0, 1.0])

    expected_s_r = s_i_hat - 2.0 * jnp.dot(n, s_i_hat) * n
    chex.assert_trees_all_close(s_r_hat, expected_s_r, atol=1e-6)


def test_eq_3_24_perpendicular_polarization_equality(
    flat_plate_scene: Scene,
) -> None:
    """Eq. (3.24), p. 80: Invariance of perpendicular polarization vector (e_perp^i = e_perp^r)."""
    paths = flat_plate_scene.trace_paths(order=1, solver="exhaustive").masked()
    v = paths.vertices[paths.mask][0]
    s_i_hat = (v[1] - v[0]) / jnp.linalg.norm(v[1] - v[0])
    s_r_hat = (v[2] - v[1]) / jnp.linalg.norm(v[2] - v[1])
    n = jnp.array([0.0, 0.0, 1.0])

    (e_i_s, _), (e_r_s, _) = sp_directions(
        s_i_hat[None, :], s_r_hat[None, :], n[None, :]
    )
    chex.assert_trees_all_close(e_i_s[0], e_r_s[0], atol=1e-6)


def test_eq_3_33_pec_tangential_electric_field_boundary_condition(
    flat_plate_scene: Scene,
) -> None:
    """Eq. (3.33), p. 82: Vanishing tangential electric field on PEC surface n x (E^i + E^r) = 0."""
    paths = flat_plate_scene.trace_paths(order=1, solver="exhaustive").masked()
    v = paths.vertices[paths.mask][0]
    s_i_hat = (v[1] - v[0]) / jnp.linalg.norm(v[1] - v[0])
    s_r_hat = (v[2] - v[1]) / jnp.linalg.norm(v[2] - v[1])
    n = jnp.array([0.0, 0.0, 1.0])

    (e_i_s, e_i_p), (e_r_s, e_r_p) = sp_directions(
        s_i_hat[None, :], s_r_hat[None, :], n[None, :]
    )
    E_i = 3.0 * e_i_p[0] + 4.0 * e_i_s[0]
    E_r = 3.0 * e_r_p[0] - 4.0 * e_r_s[0]

    total_E_tangential = jnp.cross(n, E_i + E_r)
    chex.assert_trees_all_close(total_E_tangential, jnp.zeros(3), atol=1e-6)


def test_eqs_3_36_3_37_parallel_and_perpendicular_pec_reflection() -> None:
    """Eqs. (3.36)-(3.37), p. 83: Parallel and perpendicular field components for PEC reflection."""
    cos_theta = jnp.cos(jnp.radians(30.0))
    n_r = 1e6 + 0j
    r_s, r_p = reflection_coefficients(n_r, cos_theta)
    chex.assert_trees_all_close(r_p, 1.0 + 0j, atol=1e-5)
    chex.assert_trees_all_close(r_s, -1.0 + 0j, atol=1e-5)


def test_eqs_3_38_3_39_dyadic_reflection_coefficient() -> None:
    """Eqs. (3.38)-(3.39), p. 83: Dyadic reflection coefficient matrix [R] = diag(R_h, R_s)."""
    R_h = 1.0
    R_s = -1.0
    R_mat = jnp.array([[R_h, 0.0], [0.0, R_s]])

    E_in = jnp.array([2.5, -1.5])
    E_out = R_mat @ E_in
    chex.assert_trees_all_close(E_out, jnp.array([2.5, 1.5]))


def test_eq_3_49_physical_optics_surface_current(
    flat_plate_scene: Scene,
) -> None:
    """Eq. (3.49), p. 87: Physical optics / GO surface current J_go = 2 n x H^i."""
    paths = flat_plate_scene.trace_paths(order=1, solver="exhaustive").masked()
    v = paths.vertices[paths.mask][0]
    s_i_hat = (v[1] - v[0]) / jnp.linalg.norm(v[1] - v[0])
    s_r_hat = (v[2] - v[1]) / jnp.linalg.norm(v[2] - v[1])
    n = jnp.array([0.0, 0.0, 1.0])

    (e_i_s, e_i_p), (e_r_s, e_r_p) = sp_directions(
        s_i_hat[None, :], s_r_hat[None, :], n[None, :]
    )
    E_i = 2.0 * e_i_p[0] + 5.0 * e_i_s[0]
    E_r = 2.0 * e_r_p[0] - 5.0 * e_r_s[0]

    H_i = jnp.cross(s_i_hat, E_i) / z_0
    H_r = jnp.cross(s_r_hat, E_r) / z_0

    J_go_total = jnp.cross(n, H_i + H_r)
    J_go_po = 2.0 * jnp.cross(n, H_i)
    chex.assert_trees_all_close(J_go_total, J_go_po, atol=1e-6)


def test_eqs_3_120_to_3_132_line_source_flat_strip_reflection() -> None:
    """Eqs. (3.120)-(3.132), pp. 116-118: Near-zone reflection of line source above flat conducting strip."""
    d = 2.0
    w = 6.0
    x_o = 0.5
    y_o = 3.0

    A = x_o**2 - 2.0 * d * x_o
    B = 2.0 * d**2 * y_o
    C = -(d**2) * y_o**2
    discriminant = B**2 - 4.0 * A * C
    y_r_1 = (-B + jnp.sqrt(discriminant)) / (2.0 * A)
    y_r_2 = (-B - jnp.sqrt(discriminant)) / (2.0 * A)
    y_r = jnp.where(jnp.abs(y_r_1) <= w / 2.0, y_r_1, y_r_2)

    vertices = jnp.array([
        [d, -w / 2.0, -5.0],
        [d, w / 2.0, -5.0],
        [d, w / 2.0, 5.0],
        [d, -w / 2.0, 5.0],
    ])
    triangles = jnp.array([[0, 1, 2], [0, 2, 3]])
    mesh = Mesh(
        vertices=vertices,
        triangles=triangles,
        face_materials=jnp.array([0, 0]),
        material_names=("Metal",),
    )
    scene = Scene(
        transmitters=jnp.array([0.0, 0.0, 0.0]),
        receivers=jnp.array([x_o, y_o, 0.0]),
        mesh=mesh,
    )
    paths = scene.trace_paths(order=1).masked()
    assert paths.num_valid_paths == 1
    v = paths.vertices[paths.mask][0]
    chex.assert_trees_all_close(v[1, 1], y_r, atol=1e-5)

    s_i = jnp.linalg.norm(v[1] - v[0])
    s_r = jnp.linalg.norm(v[2] - v[1])
    image_dist = jnp.sqrt((2.0 * d - x_o) ** 2 + y_o**2)
    chex.assert_trees_all_close(s_i + s_r, image_dist, atol=1e-5)


def test_eqs_3_203_3_204_eulers_curvature_formula_in_oblique_plane() -> None:
    """Eqs. (3.203)-(3.204), p. 147: Euler's theorem for surface curvatures in oblique plane."""
    a1 = 12.0
    a2 = 4.0
    alpha = jnp.radians(30.0)

    inv_a_t = jnp.sin(alpha) ** 2 / a1 + jnp.cos(alpha) ** 2 / a2
    inv_a_b = jnp.cos(alpha) ** 2 / a1 + jnp.sin(alpha) ** 2 / a2

    chex.assert_trees_all_close(1.0 / inv_a_t, 1.0 / (0.25 / 12.0 + 0.75 / 4.0))
    chex.assert_trees_all_close(1.0 / inv_a_b, 1.0 / (0.75 / 12.0 + 0.25 / 4.0))


def test_eqs_3_213_3_214_reflected_curvature_coincident_principal_plane() -> None:
    """Eqs. (3.213)-(3.214), p. 150: Reflected wavefront principal radii when alpha = 0."""
    rho_1_i = 8.0
    rho_2_i = 15.0
    a1 = 10.0
    a2 = 20.0
    theta_i = jnp.radians(45.0)

    inv_rho_1_r = 1.0 / rho_1_i + 2.0 * jnp.cos(theta_i) / a1
    inv_rho_2_r = 1.0 / rho_2_i + 2.0 / (a2 * jnp.cos(theta_i))

    chex.assert_trees_all_close(
        inv_rho_1_r, 1.0 / 8.0 + 2.0 * jnp.cos(jnp.radians(45.0)) / 10.0
    )
    chex.assert_trees_all_close(
        inv_rho_2_r, 1.0 / 15.0 + 2.0 / (20.0 * jnp.cos(jnp.radians(45.0)))
    )


def test_eqs_3_215_3_216_spherical_wave_reflection_curved_surface() -> None:
    """Eqs. (3.215)-(3.216), p. 150: Reflected wavefront radii for spherical wave when alpha = 0."""
    s_i = 10.0
    a1 = 6.0
    a2 = 12.0
    theta_i = jnp.radians(30.0)

    inv_rho_1_r = 1.0 / s_i + 2.0 * jnp.cos(theta_i) / a1
    inv_rho_2_r = 1.0 / s_i + 2.0 / (a2 * jnp.cos(theta_i))

    chex.assert_trees_all_close(
        1.0 / inv_rho_1_r,
        1.0 / (1.0 / 10.0 + 2.0 * jnp.cos(jnp.radians(30.0)) / 6.0),
    )
    chex.assert_trees_all_close(
        1.0 / inv_rho_2_r,
        1.0 / (1.0 / 10.0 + 2.0 / (12.0 * jnp.cos(jnp.radians(30.0)))),
    )


def test_eqs_3_217_3_218_plane_wave_reflection_curved_surface() -> None:
    """Eqs. (3.217)-(3.218), p. 151: Plane-wave reflection from curved surface with alpha = 0."""
    a1 = 5.0
    a2 = 15.0
    theta_i = jnp.radians(60.0)

    rho_1_r = a1 / (2.0 * jnp.cos(theta_i))
    rho_2_r = a2 * jnp.cos(theta_i) / 2.0

    chex.assert_trees_all_close(rho_1_r, 5.0 / (2.0 * 0.5))
    chex.assert_trees_all_close(rho_2_r, 15.0 * 0.5 / 2.0)


def test_eqs_3_223_to_3_227_spherical_wave_reflection_general_curved_surface() -> None:
    """Eqs. (3.223)-(3.227), pp. 151-152: Reflected radii for spherical wave on general curved surface."""
    s_i = 10.0
    a1 = 8.0
    a2 = 14.0
    alpha = jnp.radians(35.0)
    theta_i = jnp.radians(40.0)

    sin_2_theta_1 = jnp.cos(alpha) ** 2 + jnp.sin(alpha) ** 2 * jnp.cos(theta_i) ** 2
    sin_2_theta_2 = jnp.sin(alpha) ** 2 + jnp.cos(alpha) ** 2 * jnp.cos(theta_i) ** 2

    term = (sin_2_theta_2 / a1 + sin_2_theta_1 / a2) / jnp.cos(theta_i)
    radical = jnp.sqrt(term**2 - 4.0 / (a1 * a2))
    f1 = 1.0 / (term + radical)
    f2 = 1.0 / (term - radical)

    inv_rho_1_r = 1.0 / s_i + 1.0 / f1
    inv_rho_2_r = 1.0 / s_i + 1.0 / f2

    assert inv_rho_1_r > 0
    assert inv_rho_2_r > 0


def test_eqs_3_228_3_229_plane_wave_reflection_general_curved_surface() -> None:
    """Eqs. (3.228)-(3.229), p. 152: Plane-wave reflection from general curved surface (rho_{1,2}^r = f_{1,2})."""
    term = (1.0 / 10.0 + 1.0 / 10.0) / 1.0
    f1 = 1.0 / term
    f2 = 1.0 / term

    chex.assert_trees_all_close(f1, 5.0)
    chex.assert_trees_all_close(f2, 5.0)


def test_eq_6_1_kellers_law_of_edge_diffraction(wedge_scene: Scene) -> None:
    """Eq. (6.1), p. 263: Keller's law of edge diffraction s' . e = s . e (beta_0' = beta_0)."""
    paths = wedge_scene.trace_paths(
        order=1,
        solver="exhaustive",
        allowed_interactions=frozenset({InteractionType.DIFFRACTION}),
    ).masked()
    assert paths.num_valid_paths > 0
    v = paths.vertices[paths.mask][0]
    s_prime = (v[1] - v[0]) / jnp.linalg.norm(v[1] - v[0])
    s_d = (v[2] - v[1]) / jnp.linalg.norm(v[2] - v[1])
    e_hat = jnp.array([0.0, 1.0, 0.0])

    chex.assert_trees_all_close(jnp.dot(s_prime, e_hat), jnp.dot(s_d, e_hat), atol=1e-5)


def test_eqs_6_2_to_6_5_edge_fixed_basis_unit_vectors(wedge_scene: Scene) -> None:
    """Eqs. (6.2)-(6.5), p. 265: Edge-fixed coordinate system unit vectors."""
    paths = wedge_scene.trace_paths(
        order=1,
        solver="exhaustive",
        allowed_interactions=frozenset({InteractionType.DIFFRACTION}),
    ).masked()
    v = paths.vertices[paths.mask][0]
    s_prime = (v[1] - v[0]) / jnp.linalg.norm(v[1] - v[0])
    s_d = (v[2] - v[1]) / jnp.linalg.norm(v[2] - v[1])
    e_hat = jnp.array([0.0, 1.0, 0.0])

    phi_prime = normalize(-jnp.cross(e_hat, s_prime))[0]
    beta_0_prime = normalize(jnp.cross(phi_prime, s_prime))[0]
    phi_d = normalize(jnp.cross(e_hat, s_d))[0]
    beta_0_d = normalize(jnp.cross(phi_d, s_d))[0]

    chex.assert_trees_all_close(jnp.dot(phi_prime, beta_0_prime), 0.0, atol=1e-6)
    chex.assert_trees_all_close(jnp.dot(phi_prime, s_prime), 0.0, atol=1e-6)
    chex.assert_trees_all_close(jnp.dot(beta_0_prime, s_prime), 0.0, atol=1e-6)
    chex.assert_trees_all_close(jnp.dot(phi_d, beta_0_d), 0.0, atol=1e-6)
    chex.assert_trees_all_close(jnp.dot(phi_d, s_d), 0.0, atol=1e-6)
    chex.assert_trees_all_close(jnp.dot(beta_0_d, s_d), 0.0, atol=1e-6)


def test_eqs_6_6_6_7_transverse_path_components(wedge_scene: Scene) -> None:
    """Eqs. (6.6)-(6.7), p. 266: Transverse path components perpendicular to the edge."""
    paths = wedge_scene.trace_paths(
        order=1,
        solver="exhaustive",
        allowed_interactions=frozenset({InteractionType.DIFFRACTION}),
    ).masked()
    v = paths.vertices[paths.mask][0]
    s_prime = (v[1] - v[0]) / jnp.linalg.norm(v[1] - v[0])
    s_d = (v[2] - v[1]) / jnp.linalg.norm(v[2] - v[1])
    e_hat = jnp.array([0.0, 1.0, 0.0])

    s_t_prime = jnp.linalg.norm(s_prime - jnp.dot(s_prime, e_hat) * e_hat)
    s_t_d = jnp.linalg.norm(s_d - jnp.dot(s_d, e_hat) * e_hat)

    sin_beta_0_prime = jnp.sqrt(1.0 - jnp.dot(s_prime, e_hat) ** 2)
    sin_beta_0_d = jnp.sqrt(1.0 - jnp.dot(s_d, e_hat) ** 2)

    chex.assert_trees_all_close(s_t_prime, sin_beta_0_prime, atol=1e-6)
    chex.assert_trees_all_close(s_t_d, sin_beta_0_d, atol=1e-6)


def test_eq_6_8_wedge_face_tangent_vector() -> None:
    """Eq. (6.8), p. 267: Tangential unit vector on the 0-face t_o = n_o x e."""
    n_o = jnp.array([0.0, 0.0, 1.0])
    e_hat = jnp.array([0.0, 1.0, 0.0])
    t_o = jnp.cross(n_o, e_hat)
    chex.assert_trees_all_close(t_o, jnp.array([-1.0, 0.0, 0.0]))


def test_eqs_6_9_6_10_transverse_unit_vectors(wedge_scene: Scene) -> None:
    """Eqs. (6.9)-(6.10), p. 267: Transverse unit direction vectors."""
    paths = wedge_scene.trace_paths(
        order=1,
        solver="exhaustive",
        allowed_interactions=frozenset({InteractionType.DIFFRACTION}),
    ).masked()
    v = paths.vertices[paths.mask][0]
    s_prime = (v[1] - v[0]) / jnp.linalg.norm(v[1] - v[0])
    s_d = (v[2] - v[1]) / jnp.linalg.norm(v[2] - v[1])
    e_hat = jnp.array([0.0, 1.0, 0.0])

    s_t_prime_hat = normalize(s_prime - jnp.dot(s_prime, e_hat) * e_hat)[0]
    s_t_d_hat = normalize(s_d - jnp.dot(s_d, e_hat) * e_hat)[0]

    chex.assert_trees_all_close(jnp.linalg.norm(s_t_prime_hat), 1.0, atol=1e-6)
    chex.assert_trees_all_close(jnp.linalg.norm(s_t_d_hat), 1.0, atol=1e-6)
    chex.assert_trees_all_close(jnp.dot(s_t_prime_hat, e_hat), 0.0, atol=1e-6)
    chex.assert_trees_all_close(jnp.dot(s_t_d_hat, e_hat), 0.0, atol=1e-6)


def test_eq_6_11_incident_angle_from_o_face(wedge_scene: Scene) -> None:
    """Eq. (6.11), p. 267: Incident angle phi' measured from o-face."""
    paths = wedge_scene.trace_paths(
        order=1,
        solver="exhaustive",
        allowed_interactions=frozenset({InteractionType.DIFFRACTION}),
    ).masked()
    v = paths.vertices[paths.mask][0]
    s_prime = (v[1] - v[0]) / jnp.linalg.norm(v[1] - v[0])
    e_hat = jnp.array([0.0, 1.0, 0.0])
    n_o = jnp.array([0.0, 0.0, 1.0])
    t_o = jnp.cross(n_o, e_hat)

    s_t_prime_hat = normalize(s_prime - jnp.dot(s_prime, e_hat) * e_hat)[0]
    dot_t = jnp.dot(-s_t_prime_hat, t_o)
    dot_n = jnp.dot(-s_t_prime_hat, n_o)
    sgn_n = jnp.where(dot_n >= 0, 1.0, -1.0)
    phi_prime = jnp.pi - (jnp.pi - jnp.arccos(jnp.clip(dot_t, -1.0, 1.0))) * sgn_n

    assert 0 <= phi_prime <= 1.5 * jnp.pi


def test_eq_6_12_diffracted_angle_from_o_face(wedge_scene: Scene) -> None:
    """Eq. (6.12), p. 267: Diffracted angle phi measured from o-face."""
    paths = wedge_scene.trace_paths(
        order=1,
        solver="exhaustive",
        allowed_interactions=frozenset({InteractionType.DIFFRACTION}),
    ).masked()
    v = paths.vertices[paths.mask][0]
    s_d = (v[2] - v[1]) / jnp.linalg.norm(v[2] - v[1])
    e_hat = jnp.array([0.0, 1.0, 0.0])
    n_o = jnp.array([0.0, 0.0, 1.0])
    t_o = jnp.cross(n_o, e_hat)

    s_t_d_hat = normalize(s_d - jnp.dot(s_d, e_hat) * e_hat)[0]
    dot_t = jnp.dot(s_t_d_hat, t_o)
    dot_n = jnp.dot(s_t_d_hat, n_o)
    sgn_n = jnp.where(dot_n >= 0, 1.0, -1.0)
    phi = jnp.pi - (jnp.pi - jnp.arccos(jnp.clip(dot_t, -1.0, 1.0))) * sgn_n

    assert 0 <= phi <= 1.5 * jnp.pi


def test_eq_6_13_edge_fixed_diffraction_matrix_equation(
    wedge_scene: Scene,
) -> None:
    """Eq. (6.13), p. 268: Diffracted field matrix equation in edge-fixed basis."""
    paths = wedge_scene.trace_paths(
        order=1,
        solver="exhaustive",
        allowed_interactions=frozenset({InteractionType.DIFFRACTION}),
    ).masked()
    v = paths.vertices[paths.mask][0]
    s_prime = jnp.linalg.norm(v[1] - v[0])
    s_d = jnp.linalg.norm(v[2] - v[1])
    k_in = (v[1] - v[0]) / s_prime
    e_hat = jnp.array([0.0, 1.0, 0.0])

    sin_beta_0 = jnp.sqrt(1.0 - jnp.dot(k_in, e_hat) ** 2)
    rho = s_prime
    spreading = jnp.sqrt(rho / (s_d * (s_d + rho)))

    k = 2.0 * jnp.pi * 1e9 / 299792458.0
    L = (s_d * s_prime) / (s_d + s_prime) * sin_beta_0**2
    n = 1.5
    phi_prime = 0.5
    phi = 2.0
    D_s, D_h = diffraction_coefficients(k, n, phi_prime, phi, L, sin_beta_0=sin_beta_0)

    E_i_beta = 1.0
    E_i_phi = 2.0
    E_d_beta = -D_s * E_i_beta * spreading * jnp.exp(-1j * k * s_d)
    E_d_phi = -D_h * E_i_phi * spreading * jnp.exp(-1j * k * s_d)

    diff_mat = jnp.array([[-D_s, 0.0], [0.0, -D_h]])
    E_d_vec = (
        diff_mat @ jnp.array([E_i_beta, E_i_phi]) * spreading * jnp.exp(-1j * k * s_d)
    )

    chex.assert_trees_all_close(E_d_vec[0], E_d_beta, atol=1e-5)
    chex.assert_trees_all_close(E_d_vec[1], E_d_phi, atol=1e-5)


def test_eqs_6_16_6_17_incident_field_edge_fixed_decomposition() -> None:
    """Eqs. (6.16)-(6.17), p. 268: Decomposing incident electric field into edge-fixed components."""
    beta_0_prime = jnp.array([0.0, 1.0, 0.0])
    phi_prime = jnp.array([1.0, 0.0, 0.0])
    E_i = jnp.array([3.0, 4.0, 0.0])

    E_beta = jnp.dot(E_i, beta_0_prime)
    E_phi = jnp.dot(E_i, phi_prime)

    chex.assert_trees_all_close(E_beta, 4.0)
    chex.assert_trees_all_close(E_phi, 3.0)
    chex.assert_trees_all_close(E_beta * beta_0_prime + E_phi * phi_prime, E_i)


def test_eq_6_19_dyadic_diffraction_coefficient() -> None:
    """Eq. (6.19), p. 269: Dyadic diffraction coefficient D = -beta' beta D_s - phi' phi D_h."""
    D_s = 0.5 - 0.2j
    D_h = 0.8 + 0.1j
    beta_prime = jnp.array([0.0, 1.0, 0.0])
    beta_d = jnp.array([0.0, -1.0, 0.0])
    phi_prime = jnp.array([1.0, 0.0, 0.0])
    phi_d = jnp.array([-1.0, 0.0, 0.0])

    D_dyad = -D_s * jnp.outer(beta_prime, beta_d) - D_h * jnp.outer(phi_prime, phi_d)

    E_i = 2.0 * beta_prime + 3.0 * phi_prime
    E_d = E_i @ D_dyad

    chex.assert_trees_all_close(E_d, -2.0 * D_s * beta_d - 3.0 * D_h * phi_d)


def cot_times_f(
    beta: Float[Array, " *#batch"],
    n: Float[Array, " *#batch"],
    k: Float[Array, " *#batch"],
    l_val: Float[Array, " *#batch"],
    mode: Literal["+", "-"],
) -> Complex[Array, " *batch"]:
    r"""Recreate the UTD transition term :math:`\cot(x) F(k L a)`.

    Args:
        beta: The angular difference (:math:`\phi - \phi'` or :math:`\phi + \phi'`).
        n: The wedge parameter (:math:`n\pi` is the exterior angle).
        k: The wavenumber.
        l_val: The UTD distance parameter.
        mode: Sign selector, either ``"+"`` or ``"-"``.

    Returns:
        The evaluated transition term.
    """
    mode_sign = 1.0 if mode == "+" else -1.0
    N = jnp.round((beta + mode_sign * jnp.pi) / (2.0 * n * jnp.pi))
    a = 2.0 * jnp.cos(0.5 * (2.0 * n * jnp.pi * N - beta)) ** 2
    x = (jnp.pi + mode_sign * beta) / (2.0 * n)
    return (1.0 / jnp.tan(x)) * F(k * l_val * a)


def test_eq_6_20_utd_diffraction_coefficients_sum() -> None:
    """Eq. (6.20), p. 269: UTD diffraction coefficients D_{s,h} = D_1 + D_2 -/+ (D_3 + D_4)."""
    k = 50.0
    n = 1.5
    phi_prime = 0.6
    phi = 1.2
    L = 5.0
    sin_beta_0 = 1.0

    phi_minus = phi - phi_prime
    phi_plus = phi + phi_prime

    factor = -jnp.exp(-1j * jnp.pi / 4) / (
        2.0 * n * jnp.sqrt(2.0 * jnp.pi * k) * sin_beta_0
    )
    D1 = -cot_times_f(phi_minus, n, k, L, "+")
    D2 = -cot_times_f(phi_minus, n, k, L, "-")
    D3 = cot_times_f(phi_plus, n, k, L, "+")
    D4 = cot_times_f(phi_plus, n, k, L, "-")

    D_s_expected = (D1 + D2 - (D3 + D4)) * factor
    D_h_expected = (D1 + D2 + (D3 + D4)) * factor

    D_s, D_h = diffraction_coefficients(k, n, phi_prime, phi, L, sin_beta_0=sin_beta_0)
    chex.assert_trees_all_close(D_s, D_s_expected, atol=1e-5)
    chex.assert_trees_all_close(D_h, D_h_expected, atol=1e-5)


def test_eqs_6_21_to_6_24_diffraction_terms_d1_to_d4() -> None:
    """Eqs. (6.21)-(6.24), pp. 269-270: The four diffraction coefficient components D1, D2, D3, D4."""
    k = 100.0
    n = 1.5
    phi_prime = 0.5
    phi = 1.5
    L = 8.0

    beta_minus = phi - phi_prime
    beta_plus = phi + phi_prime

    # D1: Eq. (6.21)
    N1 = jnp.round((beta_minus + jnp.pi) / (2.0 * n * jnp.pi))
    a1 = 2.0 * jnp.cos(0.5 * (2.0 * n * jnp.pi * N1 - beta_minus)) ** 2
    x1 = (jnp.pi + beta_minus) / (2.0 * n)
    D1_eq = (1.0 / jnp.tan(x1)) * F(k * L * a1)
    chex.assert_trees_all_close(cot_times_f(beta_minus, n, k, L, "+"), D1_eq)

    # D2: Eq. (6.22)
    N2 = jnp.round((beta_minus - jnp.pi) / (2.0 * n * jnp.pi))
    a2 = 2.0 * jnp.cos(0.5 * (2.0 * n * jnp.pi * N2 - beta_minus)) ** 2
    x2 = (jnp.pi - beta_minus) / (2.0 * n)
    D2_eq = (1.0 / jnp.tan(x2)) * F(k * L * a2)
    chex.assert_trees_all_close(cot_times_f(beta_minus, n, k, L, "-"), D2_eq)

    # D3: Eq. (6.23)
    N3 = jnp.round((beta_plus + jnp.pi) / (2.0 * n * jnp.pi))
    a3 = 2.0 * jnp.cos(0.5 * (2.0 * n * jnp.pi * N3 - beta_plus)) ** 2
    x3 = (jnp.pi + beta_plus) / (2.0 * n)
    D3_eq = (1.0 / jnp.tan(x3)) * F(k * L * a3)
    chex.assert_trees_all_close(cot_times_f(beta_plus, n, k, L, "+"), D3_eq)

    # D4: Eq. (6.24)
    N4 = jnp.round((beta_plus - jnp.pi) / (2.0 * n * jnp.pi))
    a4 = 2.0 * jnp.cos(0.5 * (2.0 * n * jnp.pi * N4 - beta_plus)) ** 2
    x4 = (jnp.pi - beta_plus) / (2.0 * n)
    D4_eq = (1.0 / jnp.tan(x4)) * F(k * L * a4)
    chex.assert_trees_all_close(cot_times_f(beta_plus, n, k, L, "-"), D4_eq)


def test_eq_6_25_general_astigmatic_distance_parameter() -> None:
    """Eq. (6.25), p. 270: Astigmatic distance parameter L^i for general incident wavefront."""
    s_d = jnp.array(15.0)
    sin_2_beta = jnp.array(0.75)
    rho_1_i = jnp.array(8.0)
    rho_2_i = jnp.array(12.0)
    rho_e_i = jnp.array(10.0)

    expected = (
        s_d
        * (rho_e_i + s_d)
        * rho_1_i
        * rho_2_i
        * sin_2_beta
        / (rho_e_i * (rho_1_i + s_d) * (rho_2_i + s_d))
    )
    got = L_i(s_d, sin_2_beta, rho_1_i=rho_1_i, rho_2_i=rho_2_i, rho_e_i=rho_e_i)
    chex.assert_trees_all_close(got, expected)


def test_eq_6_26_spherical_wave_distance_parameter() -> None:
    """Eq. (6.26), p. 270: Distance parameter L^i for incident spherical wavefront."""
    s_d = jnp.array(20.0)
    s_i = jnp.array(10.0)
    sin_2_beta = jnp.array(0.64)

    expected = (s_d * s_i) / (s_d + s_i) * sin_2_beta
    got = L_i(s_d, sin_2_beta, s_i=s_i)
    chex.assert_trees_all_close(got, expected)


def test_eq_6_27_plane_wave_distance_parameter() -> None:
    """Eq. (6.27), p. 270: Distance parameter L^i for incident plane wave."""
    s_d = jnp.array(25.0)
    sin_2_beta = jnp.array(0.5)

    expected = s_d * sin_2_beta
    got = L_i(s_d, sin_2_beta)
    chex.assert_trees_all_close(got, expected)


def test_eq_6_28_reflection_shadow_boundary_distance_parameter() -> None:
    """Eq. (6.28), p. 271: Distance parameters L^{ro,n} associated with reflection shadow boundaries."""
    s_d = jnp.array(18.0)
    sin_2_beta = jnp.array(0.81)
    rho_1_r = jnp.array(14.0)
    rho_2_r = jnp.array(9.0)
    rho_e_r = jnp.array(11.0)

    expected = (
        s_d
        * (rho_e_r + s_d)
        * rho_1_r
        * rho_2_r
        * sin_2_beta
        / (rho_e_r * (rho_1_r + s_d) * (rho_2_r + s_d))
    )
    got = L_i(s_d, sin_2_beta, rho_1_i=rho_1_r, rho_2_i=rho_2_r, rho_e_i=rho_e_r)
    chex.assert_trees_all_close(got, expected)


def test_eq_6_34_edge_caustic_distance_curved_edge() -> None:
    """Eq. (6.34), p. 272: Edge caustic distance rho for a curved edge."""
    rho_e_i = 10.0
    a_e = 5.0
    sin_beta_0 = jnp.sin(jnp.radians(60.0))
    n_e = jnp.array([1.0, 0.0, 0.0])
    s_prime = jnp.array([0.0, 0.5, 0.8660254])
    s_d = jnp.array([0.5, 0.5, 0.7071068])

    inv_rho = 1.0 / rho_e_i - jnp.dot(n_e, s_prime - s_d) / (
        jnp.abs(a_e) * sin_beta_0**2
    )
    rho = 1.0 / inv_rho

    expected = 1.0 / (0.1 - (-0.5) / (5.0 * 0.75))
    chex.assert_trees_all_close(rho, expected)


def test_eq_6_36_straight_edge_caustic_and_spreading_factor(
    wedge_scene: Scene,
) -> None:
    """Eq. (6.36), p. 273: Straight edge caustic distance rho = rho_e^i and spreading factor."""
    paths = wedge_scene.trace_paths(
        order=1,
        solver="exhaustive",
        allowed_interactions=frozenset({InteractionType.DIFFRACTION}),
    ).masked()
    v = paths.vertices[paths.mask][0]
    s_prime = jnp.linalg.norm(v[1] - v[0])
    s_d = jnp.linalg.norm(v[2] - v[1])

    spreading_eq_6_36 = jnp.sqrt(s_prime / (s_prime + s_d)) / jnp.sqrt(s_d)

    solver = GeometricFieldSolver()
    solver_spreading = solver.spreading_factor(paths, wedge_scene.mesh)[paths.mask][0]
    chex.assert_trees_all_close(solver_spreading, spreading_eq_6_36 / s_prime)


def test_eq_6_37_cylindrical_wave_diffraction_spreading_factor(
    wedge_scene: Scene,
) -> None:
    """Eq. (6.37), p. 273: Straight edge diffraction spreading factor 1/sqrt(s) for plane wave."""
    paths = wedge_scene.trace_paths(
        order=1,
        solver="exhaustive",
        allowed_interactions=frozenset({InteractionType.DIFFRACTION}),
    ).masked()
    v = paths.vertices[paths.mask][0]
    s_d = jnp.linalg.norm(v[2] - v[1])

    solver = GeometricFieldSolver(tx_wavefront_radii=None)
    spreading = solver.spreading_factor(paths, wedge_scene.mesh)[paths.mask][0]
    chex.assert_trees_all_close(spreading, 1.0 / jnp.sqrt(s_d))


def test_eq_6_38_far_zone_edge_spreading_factor() -> None:
    """Eq. (6.38), p. 273: Far-zone edge spreading factor approximation sqrt(rho)/s when s >> rho."""
    rho = 5.0
    s = 1e5

    exact_spreading = jnp.sqrt(rho / (s * (s + rho)))
    far_zone_approx = jnp.sqrt(rho) / s

    chex.assert_trees_all_close(exact_spreading, far_zone_approx, rtol=1e-4)
