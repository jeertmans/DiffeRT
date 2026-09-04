import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from differt.em import (
    InteractionType,
    PathWavefront,
    WavefrontState,
    propagate_wavefront,
)
from differt.geometry import Mesh, TracedPaths


class TestWavefrontState:
    def test_from_tx_default_spherical(self) -> None:
        k_hat = jnp.array([1.0, 0.0, 0.0])
        state = WavefrontState.from_tx(k_hat, tx_wavefront=0.0)
        chex.assert_trees_all_close(state.radii, jnp.array([0.0, 0.0]))
        assert state.axes.shape == (2, 3)
        # Verify axes are orthogonal and perpendicular to k_hat
        chex.assert_trees_all_close(
            jnp.dot(state.axes[0], state.axes[1]), 0.0, atol=1e-6
        )
        chex.assert_trees_all_close(jnp.dot(state.axes[0], k_hat), 0.0, atol=1e-6)
        chex.assert_trees_all_close(jnp.dot(state.axes[1], k_hat), 0.0, atol=1e-6)
        chex.assert_trees_all_equal(state.is_planar, jnp.array([False, False]))

    def test_from_tx_planar(self) -> None:
        k_hat = jnp.array([0.0, 0.0, 1.0])
        state = WavefrontState.from_tx(k_hat, tx_wavefront=None)
        chex.assert_trees_all_equal(state.is_planar, jnp.array([True, True]))
        chex.assert_trees_all_close(state.radii, jnp.array([0.0, 0.0]))

    def test_from_tx_astigmatic_tuple_2(self) -> None:
        k_hat = jnp.array([0.0, 1.0, 0.0])
        state = WavefrontState.from_tx(k_hat, tx_wavefront=(2.0, 5.0))
        chex.assert_trees_all_close(state.radii, jnp.array([2.0, 5.0]))
        chex.assert_trees_all_equal(state.is_planar, jnp.array([False, False]))

    def test_from_tx_astigmatic_tuple_4(self) -> None:
        k_hat = jnp.array([0.0, 0.0, 1.0])
        ax_s = jnp.array([1.0, 0.0, 0.0])
        ax_p = jnp.array([0.0, 1.0, 0.0])
        state = WavefrontState.from_tx(k_hat, tx_wavefront=(1.5, ax_s, 4.5, ax_p))
        chex.assert_trees_all_close(state.radii, jnp.array([1.5, 4.5]))
        chex.assert_trees_all_close(state.axes[0], ax_s)
        chex.assert_trees_all_close(state.axes[1], ax_p)
        chex.assert_trees_all_equal(state.is_planar, jnp.array([False, False]))

    def test_from_tx_existing_instance(self) -> None:
        k_hat = jnp.array([1.0, 0.0, 0.0])
        state = WavefrontState.from_tx(k_hat, tx_wavefront=3.0)
        returned = WavefrontState.from_tx(k_hat, tx_wavefront=state)
        assert returned is state

    def test_propagate(self) -> None:
        k_hat = jnp.array([1.0, 0.0, 0.0])
        spherical = WavefrontState.from_tx(k_hat, tx_wavefront=2.0)
        propagated_sph = spherical.propagate(jnp.array(3.0))
        chex.assert_trees_all_close(propagated_sph.radii, jnp.array([5.0, 5.0]))
        chex.assert_trees_all_close(propagated_sph.axes, spherical.axes)

        planar = WavefrontState.from_tx(k_hat, tx_wavefront=None)
        propagated_pl = planar.propagate(jnp.array(10.0))
        chex.assert_trees_all_close(propagated_pl.radii, jnp.array([0.0, 0.0]))
        chex.assert_trees_all_equal(propagated_pl.is_planar, jnp.array([True, True]))

    def test_reflect(self) -> None:
        k_hat = jnp.array([1.0, 0.0, 0.0])
        normal = jnp.array([-1.0, 0.0, 0.0])
        state = WavefrontState.from_tx(k_hat, tx_wavefront=(1.0, 2.0))
        reflected = state.reflect(normal)
        chex.assert_trees_all_close(reflected.radii, state.radii)
        chex.assert_trees_all_equal(reflected.is_planar, state.is_planar)
        # Unit normal reflection reverses normal component of axes
        for i in range(2):
            expected = state.axes[i] - 2.0 * jnp.dot(state.axes[i], normal) * normal
            chex.assert_trees_all_close(reflected.axes[i], expected, atol=1e-6)

    def test_transmit(self) -> None:
        k_hat = jnp.array([0.0, 0.0, 1.0])
        state = WavefrontState.from_tx(k_hat, tx_wavefront=1.0)
        transmitted = state.transmit()
        chex.assert_trees_all_close(transmitted.radii, state.radii)
        chex.assert_trees_all_close(transmitted.axes, state.axes)

    def test_diffract_spherical(self) -> None:
        # McNamara et al. (1990), Chapter 6, Section 6.3, pp. 268-273 (PDF pp. 143-145):
        # For a straight edge (a_e -> inf), Eq. (6.34) and Eq. (6.36) give rho = rho_e^i.
        # For spherical wave incidence with radius s' = 5.0, rho_1^i = rho_2^i = rho_e^i = 5.0.
        # Edge caustic along the edge has rho_1^d = 0.0, and second caustic has rho_2^d = rho_e^i = 5.0.
        k_in = jnp.array([1.0, 0.0, 0.0])
        k_out = jnp.array([0.0, 1.0, 0.0])
        e_hat = jnp.array([0.0, 0.0, 1.0])
        state = WavefrontState.from_tx(k_in, tx_wavefront=5.0)

        new_state, inc_radii = state.diffract(k_in, k_out, e_hat)
        chex.assert_trees_all_close(new_state.radii[0], 0.0)
        chex.assert_trees_all_close(new_state.radii[1], 5.0)
        chex.assert_trees_all_close(inc_radii, jnp.array([5.0, 5.0, 5.0]))
        chex.assert_trees_all_equal(new_state.is_planar, jnp.array([False, False]))

    def test_diffract_edge_fixed_coordinate_system(self) -> None:
        # McNamara et al. (1990), Chapter 6, Section 6.2, pp. 266-267 (PDF p. 142):
        # Incident ray unit vectors:
        #   phi_hat_prime = - (e_hat x s_hat_prime) / |e_hat x s_hat_prime|  (Eq. 6.2)
        #   beta_0_prime_hat = phi_hat_prime x s_hat_prime                   (Eq. 6.3)
        # Diffracted ray unit vectors:
        #   phi_hat = (e_hat x s_hat) / |e_hat x s_hat|                      (Eq. 6.4)
        #   beta_0_hat = phi_hat x s_hat                                     (Eq. 6.5)
        # Here: s_hat_prime = k_in = (1, 0, 0), s_hat = k_out = (0, 1, 0), e_hat = (0, 0, 1).
        k_in = jnp.array([1.0, 0.0, 0.0])
        k_out = jnp.array([0.0, 1.0, 0.0])
        e_hat = jnp.array([0.0, 0.0, 1.0])
        state = WavefrontState.from_tx(k_in, tx_wavefront=4.0)

        new_state, _ = state.diffract(k_in, k_out, e_hat)

        # Theoretical diffracted unit vectors from McNamara Eq. (6.4)-(6.5):
        # phi_hat = (0, 0, 1) x (0, 1, 0) / 1 = (-1, 0, 0)
        expected_phi_d = jnp.array([-1.0, 0.0, 0.0])
        # beta_0_hat = (-1, 0, 0) x (0, 1, 0) = (0, 0, -1)
        expected_beta_0_d = jnp.array([0.0, 0.0, -1.0])

        chex.assert_trees_all_close(new_state.axes[0], expected_phi_d, atol=1e-6)
        chex.assert_trees_all_close(new_state.axes[1], expected_beta_0_d, atol=1e-6)

    def test_diffract_astigmatic_euler_curvature(self) -> None:
        # McNamara et al. (1990), Chapter 6, Section 6.3, pp. 270-273 (PDF pp. 144-145):
        # Radius of curvature in edge-fixed plane of incidence follows Euler's theorem:
        #   1 / rho_e^i = (cos^2 alpha_1) / rho_1^i + (cos^2 alpha_2) / rho_2^i
        # where alpha_1, alpha_2 are angles between beta_0_prime and principal axes u_1, u_2.
        k_in = jnp.array([1.0, 0.0, 0.0])
        k_out = jnp.array([0.0, 1.0, 0.0])
        e_hat = jnp.array([0.0, 0.0, 1.0])

        # beta_0_prime is (0, 0, -1). Let principal axes be rotated by 45 degrees in y-z plane:
        theta = jnp.pi / 4.0
        u1 = jnp.array([0.0, jnp.sin(theta), jnp.cos(theta)])
        u2 = jnp.array([0.0, jnp.cos(theta), -jnp.sin(theta)])
        rho1 = 2.0
        rho2 = 6.0
        state = WavefrontState.from_tx(k_in, tx_wavefront=(rho1, u1, rho2, u2))

        new_state, inc_radii = state.diffract(k_in, k_out, e_hat)

        # cos(alpha1)^2 = (-cos(pi/4))^2 = 0.5, cos(alpha2)^2 = (sin(pi/4))^2 = 0.5
        expected_curv_e = 0.5 * (1.0 / rho1) + 0.5 * (1.0 / rho2)
        expected_rho_e = 1.0 / expected_curv_e

        chex.assert_trees_all_close(inc_radii[0], rho1)
        chex.assert_trees_all_close(inc_radii[1], rho2)
        chex.assert_trees_all_close(inc_radii[2], expected_rho_e, rtol=1e-5)
        chex.assert_trees_all_close(new_state.radii[0], 0.0)
        chex.assert_trees_all_close(new_state.radii[1], expected_rho_e, rtol=1e-5)

    def test_diffract_cylindrical_zero_curvature_edge_plane(self) -> None:
        # McNamara et al. (1990), Chapter 6, Section 6.3, p. 273 (PDF p. 145), Eq. (6.37):
        # When a cylindrical wave is incident upon the straight edge with axis parallel to the edge,
        # the curvature in the plane containing s_hat_prime and e_hat is zero (rho_e^i -> inf).
        k_in = jnp.array([1.0, 0.0, 0.0])
        k_out = jnp.array([0.0, 1.0, 0.0])
        e_hat = jnp.array([0.0, 0.0, 1.0])

        # Cylindrical wave: planar along z (u1 = (0, 0, 1)), radius 4.0 along y (u2 = (0, 1, 0)):
        u1 = jnp.array([0.0, 0.0, 1.0])
        u2 = jnp.array([0.0, 1.0, 0.0])
        state = WavefrontState(
            radii=jnp.array([0.0, 4.0]),
            axes=jnp.stack([u1, u2], axis=0),
            is_planar=jnp.array([True, False]),
        )

        new_state, inc_radii = state.diffract(k_in, k_out, e_hat)

        # beta_0_prime is along z, so curvature along beta_0_prime is 0.0 => rho_e^i = inf
        assert jnp.isinf(inc_radii[2])
        # Diffracted wave has caustic along edge (radii[0] = 0.0) and planar along beta_0_d (is_planar[1] = True)
        chex.assert_trees_all_close(new_state.radii[0], 0.0)
        assert bool(new_state.is_planar[1])

    def test_diffract_planar(self) -> None:
        # McNamara et al. (1990), Chapter 6, Section 6.3, p. 270, Eq. (6.27) & p. 273, Eq. (6.37):
        # Plane wave incidence has rho_1^i = rho_2^i = rho_e^i = inf.
        k_in = jnp.array([1.0, 0.0, 0.0])
        k_out = jnp.array([0.0, 1.0, 0.0])
        e_hat = jnp.array([0.0, 0.0, 1.0])
        state = WavefrontState.from_tx(k_in, tx_wavefront=None)

        new_state, inc_radii = state.diffract(k_in, k_out, e_hat)
        assert jnp.all(jnp.isinf(inc_radii))
        chex.assert_trees_all_close(new_state.radii[0], 0.0)
        assert not bool(new_state.is_planar[0])
        assert bool(new_state.is_planar[1])


class TestPathWavefront:
    def test_attributes_and_shapes(self) -> None:
        k_hat = jnp.array([1.0, 0.0, 0.0])
        state = WavefrontState.from_tx(k_hat, tx_wavefront=0.0)
        pw = PathWavefront(
            state=state,
            incident_radii=jnp.zeros((1, 3)),
            spreading_factor=jnp.array(0.5),
            segment_radii=jnp.zeros((2, 2)),
        )
        assert pw.state is state
        assert pw.incident_radii.shape == (1, 3)
        assert pw.spreading_factor.shape == ()
        assert pw.segment_radii.shape == (2, 2)


def test_propagate_wavefront_los() -> None:
    tx = jnp.array([0.0, 0.0, 0.0])
    rx = jnp.array([10.0, 0.0, 0.0])
    vertices = jnp.array([[tx, rx]])
    objects = jnp.array([[-1, -1]])
    mask = jnp.array([True])
    interaction_types = jnp.empty((1, 0), dtype=int)
    paths = TracedPaths(
        vertices=vertices,
        objects=objects,
        mask=mask,
        interaction_types=interaction_types,
    )
    mesh = Mesh(
        vertices=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        triangles=jnp.array([[0, 1, 2]]),
        face_materials=jnp.array([0]),
        material_names=("Metal",),
    )

    # Spherical tx: spreading is 1 / s_0 = 1 / 10 = 0.1
    pw_sph = propagate_wavefront(paths, mesh, tx_wavefront=0.0)
    chex.assert_trees_all_close(pw_sph.spreading_factor, jnp.array([0.1]), rtol=1e-5)
    chex.assert_trees_all_close(pw_sph.state.radii, jnp.array([[10.0, 10.0]]))

    # Planar tx: spreading is 1.0
    pw_pl = propagate_wavefront(paths, mesh, tx_wavefront=None)
    chex.assert_trees_all_close(pw_pl.spreading_factor, jnp.array([1.0]), rtol=1e-5)


def test_propagate_wavefront_reflection() -> None:
    tx = jnp.array([0.0, 0.0, 10.0])
    bounce = jnp.array([0.0, 0.0, 0.0])
    rx = jnp.array([0.0, 0.0, 10.0])
    vertices = jnp.array([[tx, bounce, rx]])
    objects = jnp.array([[-1, 0, -1]])
    mask = jnp.array([True])
    interaction_types = jnp.array([[InteractionType.REFLECTION]])
    paths = TracedPaths(
        vertices=vertices,
        objects=objects,
        mask=mask,
        interaction_types=interaction_types,
    )
    mesh = Mesh(
        vertices=jnp.array([
            [-100.0, -100.0, 0.0],
            [100.0, -100.0, 0.0],
            [0.0, 100.0, 0.0],
        ]),
        triangles=jnp.array([[0, 1, 2]]),
        face_materials=jnp.array([0]),
        material_names=("Metal",),
    )

    pw = propagate_wavefront(paths, mesh, tx_wavefront=0.0)
    # Total distance is 10 + 10 = 20, spreading factor = 1 / 20 = 0.05
    chex.assert_trees_all_close(pw.spreading_factor, jnp.array([0.05]), rtol=1e-5)
    chex.assert_trees_all_close(pw.state.radii, jnp.array([[20.0, 20.0]]))


def test_propagate_wavefront_diffraction() -> None:
    # McNamara et al. (1990), Chapter 6, Section 6.3, pp. 268-273 (PDF pp. 143-145):
    # Tests straight-edge diffraction spreading factors:
    # - General astigmatic wave: Eq. (6.13) and Eq. (6.36)
    # - Planar wave limit: Eq. (6.37)
    # - Spherical wave limit: Eq. (6.13) with rho = s'
    vertices_m = jnp.array([
        [0.0, -30.0, -15.0],
        [0.0, -30.0, 15.0],
        [0.0, 0.0, 15.0],
        [0.0, 0.0, -15.0],
        [30.0, 0.0, 15.0],
        [30.0, 0.0, -15.0],
    ])
    triangles_m = jnp.array([[0, 1, 2], [0, 2, 3], [3, 2, 4], [3, 4, 5]])
    mesh = Mesh(
        vertices=vertices_m,
        triangles=triangles_m,
        face_materials=jnp.array([0, 0, 0, 0]),
        material_names=("Concrete",),
    )
    v_tx = jnp.array([-10.0, -5.0, 0.0])
    v_edge = jnp.array([0.0, 0.0, 0.0])
    v_rx = jnp.array([5.0, 10.0, 0.0])
    s_prime = jnp.linalg.norm(v_edge - v_tx)
    s_after = jnp.linalg.norm(v_rx - v_edge)

    paths = TracedPaths(
        vertices=jnp.array([[v_tx, v_edge, v_rx]]),
        objects=jnp.array([[-1, 5, -1]]),
        mask=jnp.array([True]),
        interaction_types=jnp.array([[InteractionType.DIFFRACTION]]),
    )

    # 1. Astigmatic wavefront
    pw_astigmatic = propagate_wavefront(paths, mesh, tx_wavefront=(3.0, 8.0))
    assert jnp.all(jnp.isfinite(pw_astigmatic.spreading_factor))
    assert jnp.all(pw_astigmatic.spreading_factor > 0.0)
    assert pw_astigmatic.incident_radii.shape[-2:] == (1, 3)

    # 2. Planar wavefront: spreading is exactly 1 / sqrt(s_after), McNamara Eq. (6.37), p. 273
    pw_planar = propagate_wavefront(paths, mesh, tx_wavefront=None)
    expected_planar = 1.0 / jnp.sqrt(s_after)
    chex.assert_trees_all_close(
        pw_planar.spreading_factor, jnp.array([expected_planar]), rtol=1e-5
    )

    # 3. Spherical wavefront: spreading is 1 / sqrt(s_prime * s_after * (s_prime + s_after))
    pw_spherical = propagate_wavefront(paths, mesh, tx_wavefront=0.0)
    expected_spherical = 1.0 / jnp.sqrt(s_prime * s_after * (s_prime + s_after))
    chex.assert_trees_all_close(
        pw_spherical.spreading_factor, jnp.array([expected_spherical]), rtol=1e-5
    )


def test_propagate_wavefront_diffraction_differentiable() -> None:
    # Verify that reverse-mode automatic differentiation does not produce NaNs
    # across planar, spherical, and astigmatic diffracted wavefronts.
    vertices_m = jnp.array([
        [0.0, -30.0, -15.0],
        [0.0, -30.0, 15.0],
        [0.0, 0.0, 15.0],
        [0.0, 0.0, -15.0],
        [30.0, 0.0, 15.0],
        [30.0, 0.0, -15.0],
    ])
    triangles_m = jnp.array([[0, 1, 2], [0, 2, 3], [3, 2, 4], [3, 4, 5]])
    mesh = Mesh(
        vertices=vertices_m,
        triangles=triangles_m,
        face_materials=jnp.array([0, 0, 0, 0]),
        material_names=("Concrete",),
    )

    def loss(
        rx_pos: Float[Array, "3"], tx_wf: tuple[float, float] | float | None
    ) -> Float[Array, ""]:
        paths = TracedPaths(
            vertices=jnp.array([[[-10.0, -5.0, 0.0], [0.0, 0.0, 0.0], rx_pos]]),
            objects=jnp.array([[-1, 5, -1]]),
            mask=jnp.array([True]),
            interaction_types=jnp.array([[InteractionType.DIFFRACTION]]),
        )
        pw = propagate_wavefront(paths, mesh, tx_wavefront=tx_wf)
        return jnp.sum(pw.spreading_factor)

    rx_init = jnp.array([5.0, 10.0, 0.0])

    # Planar: previously caused NaN gradients due to unselected branch inf/inf
    grad_planar = jax.grad(lambda pos: loss(pos, None))(rx_init)
    assert jnp.all(jnp.isfinite(grad_planar))

    # Spherical
    grad_spherical = jax.grad(lambda pos: loss(pos, 0.0))(rx_init)
    assert jnp.all(jnp.isfinite(grad_spherical))

    # Astigmatic
    grad_astigmatic = jax.grad(lambda pos: loss(pos, (3.0, 8.0)))(rx_init)
    assert jnp.all(jnp.isfinite(grad_astigmatic))


def test_propagate_wavefront_jit() -> None:
    tx = jnp.array([0.0, 0.0, 0.0])
    rx = jnp.array([10.0, 0.0, 0.0])
    paths = TracedPaths(
        vertices=jnp.array([[tx, rx]]),
        objects=jnp.array([[-1, -1]]),
        mask=jnp.array([True]),
        interaction_types=jnp.empty((1, 0), dtype=int),
    )
    mesh = Mesh(
        vertices=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        triangles=jnp.array([[0, 1, 2]]),
        face_materials=jnp.array([0]),
        material_names=("Metal",),
    )

    jitted = jax.jit(propagate_wavefront)
    pw = jitted(paths, mesh, 0.0)
    chex.assert_trees_all_close(pw.spreading_factor, jnp.array([0.1]), rtol=1e-5)
