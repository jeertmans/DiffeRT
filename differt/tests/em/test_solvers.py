from collections.abc import Mapping
from typing import Any

import chex
import equinox as eqx
import jax.numpy as jnp
import pytest
from jaxtyping import Array, ArrayLike, Complex, Float

from differt.em import (
    AbstractFieldSolver,
    GeometricFieldSolver,
    InteractionType,
    Material,
    compute_received_fields,
    compute_received_power,
    fspl,
    materials,
    refraction_coefficients,
)
from differt.geometry import Mesh, TracedPaths


def _los_paths(tx: list[float], rx: list[float]) -> TracedPaths:
    vertices = jnp.array([[tx, rx]])
    objects = jnp.full(vertices.shape[:-1], -1, dtype=int)
    mask = jnp.ones(vertices.shape[:-2], dtype=bool)
    interaction_types = jnp.empty((*vertices.shape[:-2], 0), dtype=int)
    return TracedPaths(
        vertices=vertices,
        objects=objects,
        mask=mask,
        interaction_types=interaction_types,
    )


def _single_bounce_paths(
    tx: list[float],
    bounce: list[float],
    rx: list[float],
    interaction_type: InteractionType,
) -> TracedPaths:
    vertices = jnp.array([[tx, bounce, rx]])
    objects = jnp.array([[-1, 0, -1]])
    mask = jnp.ones(vertices.shape[:-2], dtype=bool)
    interaction_types = jnp.array([[interaction_type]])
    return TracedPaths(
        vertices=vertices,
        objects=objects,
        mask=mask,
        interaction_types=interaction_types,
    )


def _ground_plane_mesh(material_name: str = "Metal") -> Mesh:
    return Mesh(
        vertices=jnp.array([
            [-100.0, -100.0, 0.0],
            [100.0, -100.0, 0.0],
            [0.0, 100.0, 0.0],
        ]),
        triangles=jnp.array([[0, 1, 2]]),
        face_materials=jnp.array([0]),
        material_names=(material_name,),
    )


class TestGeometricFieldSolver:
    def test_default_supported_interaction_types(self) -> None:
        assert GeometricFieldSolver.supported_interaction_types == frozenset({
            InteractionType.REFLECTION,
            InteractionType.DIFFRACTION,
            InteractionType.TRANSMISSION,
            InteractionType.SCATTERING,
        })

    def test_compute_received_fields_defaults_to_geometric_field_solver(self) -> None:
        paths = _los_paths([0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        mesh = Mesh.empty()

        got = compute_received_fields(paths, mesh, 1e9)
        expected = GeometricFieldSolver().compute_fields(paths, mesh, 1e9)

        chex.assert_trees_all_close(got, expected)

    def test_custom_solver_is_used(self) -> None:
        paths = _los_paths([0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
        mesh = Mesh.empty()

        class ConstantFieldSolver(AbstractFieldSolver):
            def compute_fields(
                self,
                paths: TracedPaths,
                mesh: Mesh,  # ruff:ignore[unused-method-argument]
                frequency: Float[ArrayLike, "*#batch"],  # ruff:ignore[unused-method-argument]
                **kwargs: Any,  # ruff:ignore[unused-method-argument]
            ) -> Complex[Array, "*batch"]:
                return jnp.ones(paths.shape, dtype=complex)

        got = compute_received_fields(paths, mesh, 1e9, solver=ConstantFieldSolver())

        chex.assert_trees_all_close(got, jnp.ones(paths.shape, dtype=complex))

    def test_reflection_off_perfect_conductor_matches_fspl(self) -> None:
        # TX at (0, 0, 1), bounce off the ground plane (z=0) at (5, 0, 0), RX at (10, 0, 1).
        # 'Metal' (conductivity ~1e7 S/m) is not a *perfect* conductor, so |r| is
        # only extremely close to (not exactly) 1; hence the loose-ish tolerance
        # below. The received power should match free-space path loss over the
        # *total* (unfolded) path length, regardless of polarization.
        paths = _single_bounce_paths(
            [0.0, 0.0, 1.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 1.0],
            InteractionType.REFLECTION,
        )
        mesh = _ground_plane_mesh("Metal")
        frequency = 1e9
        total_length = 2 * jnp.sqrt(5.0**2 + 1.0**2)

        # Co-polarized only: a specular reflection in the (TX, bounce, RX) plane
        # does not couple V and H, so cross-polarized power would be near zero.
        for tx_pol, rx_pol in (("V", "V"), ("H", "H")):
            fields = compute_received_fields(
                paths, mesh, frequency, tx_polarization=tx_pol, rx_polarization=rx_pol
            )
            power_dbw = compute_received_power(fields, z_0=1.0)
            loss_db = fspl(total_length, frequency, dB=True)

            chex.assert_trees_all_close(power_dbw, -loss_db, atol=1e-2)

    def test_reflection_matrix_shape_and_dtype(self) -> None:
        paths = _single_bounce_paths(
            [0.0, 0.0, 1.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 1.0],
            InteractionType.REFLECTION,
        )
        mesh = _ground_plane_mesh("Metal")

        mat = GeometricFieldSolver().reflection_matrix(paths, mesh, 1e9, materials)

        chex.assert_shape(mat, (1, 1, 2, 2))
        assert jnp.iscomplexobj(mat)

    @pytest.mark.parametrize(
        "frequency",
        [
            jnp.array(1e9),
            jnp.array([1e9]),
        ],
        ids=["scalar", "batched"],
    )
    def test_frequency_broadcasts_against_batch(
        self, frequency: Float[Array, "*#batch"]
    ) -> None:
        paths = _single_bounce_paths(
            [0.0, 0.0, 1.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 1.0],
            InteractionType.REFLECTION,
        )
        mesh = _ground_plane_mesh("Metal")

        got = compute_received_fields(paths, mesh, frequency)
        expected = compute_received_fields(paths, mesh, 1e9)

        chex.assert_trees_all_close(got, expected)

    def test_unsupported_interaction_type_raises(self) -> None:
        class ReflectionOnlyFieldSolver(GeometricFieldSolver):
            supported_interaction_types = frozenset({InteractionType.REFLECTION})

        paths = _single_bounce_paths(
            [0.0, 0.0, 1.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 1.0],
            InteractionType.SCATTERING,
        )
        mesh = _ground_plane_mesh("Metal")

        with pytest.raises(Exception, match="does not support"):
            compute_received_fields(
                paths, mesh, 1e9, solver=ReflectionOnlyFieldSolver()
            )

    def test_unsupported_interaction_type_raises_under_jit(self) -> None:
        class ReflectionOnlyFieldSolver(GeometricFieldSolver):
            supported_interaction_types = frozenset({InteractionType.REFLECTION})

        mesh = _ground_plane_mesh("Metal")
        solver = ReflectionOnlyFieldSolver()

        @eqx.filter_jit
        def f(
            vertices: Float[Array, "*batch path_length 3"],
        ) -> Complex[Array, "*batch"]:
            objects = jnp.array([[-1, 0, -1]])
            mask = jnp.ones(vertices.shape[:-2], dtype=bool)
            interaction_types = jnp.array([[InteractionType.SCATTERING]])
            paths = TracedPaths(
                vertices=vertices,
                objects=objects,
                mask=mask,
                interaction_types=interaction_types,
            )
            return compute_received_fields(paths, mesh, 1e9, solver=solver)

        vertices = jnp.array([[[0.0, 0.0, 1.0], [5.0, 0.0, 0.0], [10.0, 0.0, 1.0]]])

        with pytest.raises(Exception, match="does not support"):
            f(vertices)

    def test_extending_with_a_new_interaction_type(self) -> None:
        # Demonstrates the general customization pattern: override the
        # matching '*_matrix' method (here with a trivial identity, in place
        # of the real UTD physics) and redeclare 'supported_interaction_types'.
        class IdentityDiffractionFieldSolver(GeometricFieldSolver):
            supported_interaction_types = frozenset({
                InteractionType.REFLECTION,
                InteractionType.DIFFRACTION,
            })

            def diffraction_matrix(
                self,
                paths: TracedPaths,
                mesh: Mesh,  # ruff:ignore[unused-method-argument]
                frequency: Float[ArrayLike, "*#batch"],  # ruff:ignore[unused-method-argument]
                radio_materials: Mapping[str, Material],  # ruff:ignore[unused-method-argument]
                tx_wavefront_radius: Float[  # ruff:ignore[unused-method-argument]
                    ArrayLike, "*#batch"
                ] = 0.0,
            ) -> Complex[Array, "*batch order 2 2"]:
                shape = (*paths.interaction_types.shape, 2, 2)
                return jnp.broadcast_to(jnp.eye(2, dtype=complex), shape)

        paths = _single_bounce_paths(
            [0.0, 0.0, 1.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 1.0],
            InteractionType.DIFFRACTION,
        )
        # 'supported_interaction_types' still includes REFLECTION (inherited), so
        # 'reflection_matrix' is still called (and discarded) for every bounce; the
        # mesh must therefore carry valid materials even though no bounce is a
        # reflection here.
        mesh = _ground_plane_mesh("Metal")
        solver = IdentityDiffractionFieldSolver()

        # Should not raise, unlike the default solver.
        fields = compute_received_fields(paths, mesh, 1e9, solver=solver)
        assert jnp.all(jnp.isfinite(fields))


class TestNonPlanarWavefront:
    """
    Validates ``tx_wavefront_radius`` (a non-planar, near-field source),
    which has no Sionna RT equivalent to cross-check against (see
    :class:`GeometricFieldSolver<differt.em.GeometricFieldSolver>`'s
    docstring note). Instead, each test checks a geometric equivalence: a
    point source with ``tx_wavefront_radius=rho0`` at some position must
    give *exactly* the same field as an ideal point source
    (``tx_wavefront_radius=0``) physically moved back by ``rho0`` along
    the direction of the first path segment -- both represent the same
    spherical wavefront arriving at the transmitter.
    """

    def test_los(self) -> None:
        mesh = Mesh.empty()
        frequency = 3.5e9
        rho0 = 2.5
        tx, rx = jnp.array([0.0, 0.0, 0.0]), jnp.array([10.0, 0.0, 0.0])

        paths = _los_paths(tx, rx)
        got = compute_received_fields(paths, mesh, frequency, tx_wavefront_radius=rho0)

        k_hat = (rx - tx) / jnp.linalg.norm(rx - tx)
        moved_paths = _los_paths(tx - rho0 * k_hat, rx)
        expected = compute_received_fields(moved_paths, mesh, frequency)

        chex.assert_trees_all_close(got, expected, rtol=1e-5)

    def test_reflection(self) -> None:
        mesh = _ground_plane_mesh("Metal")
        frequency = 3.5e9
        rho0 = 3.0
        tx = jnp.array([0.0, 0.0, 1.0])
        bounce = jnp.array([5.0, 0.0, 0.0])
        rx = jnp.array([10.0, 0.0, 1.0])

        paths = _single_bounce_paths(tx, bounce, rx, InteractionType.REFLECTION)
        got = compute_received_fields(paths, mesh, frequency, tx_wavefront_radius=rho0)

        k_hat = (bounce - tx) / jnp.linalg.norm(bounce - tx)
        moved_paths = _single_bounce_paths(
            tx - rho0 * k_hat, bounce, rx, InteractionType.REFLECTION
        )
        expected = compute_received_fields(moved_paths, mesh, frequency)

        chex.assert_trees_all_close(got, expected, rtol=1e-5)

    def test_diffraction(self) -> None:
        # Reuses the canonical 90-degree wedge from TestDiffractionAgainstSionna.
        vertices = jnp.array([
            [0.0, -30.0, -15.0],
            [0.0, -30.0, 15.0],
            [0.0, 0.0, 15.0],
            [0.0, 0.0, -15.0],
            [30.0, 0.0, 15.0],
            [30.0, 0.0, -15.0],
        ])
        triangles = jnp.array([[0, 1, 2], [0, 2, 3], [3, 2, 4], [3, 4, 5]])
        wedge = Mesh(
            vertices=vertices,
            triangles=triangles,
            face_materials=jnp.array([0, 0, 0, 0]),
            material_names=("Concrete",),
        )
        frequency = 3.5e9
        rho0 = 4.0
        tx = jnp.array([-10.0, -5.0, 0.0])
        diffraction_point = jnp.array([0.0, 0.0, 0.0])
        rx = jnp.array([5.0, 10.0, 0.0])

        def diffraction_paths(tx_pos: Float[Array, "3"]) -> TracedPaths:
            return TracedPaths(
                vertices=jnp.array([[tx_pos, diffraction_point, rx]]),
                objects=jnp.array([[-1, 5, -1]]),
                mask=jnp.array([True]),
                interaction_types=jnp.array([[InteractionType.DIFFRACTION]]),
            )

        got = compute_received_fields(
            diffraction_paths(tx), wedge, frequency, tx_wavefront_radius=rho0
        )

        k_hat = (diffraction_point - tx) / jnp.linalg.norm(diffraction_point - tx)
        expected = compute_received_fields(
            diffraction_paths(tx - rho0 * k_hat), wedge, frequency
        )

        chex.assert_trees_all_close(got, expected, rtol=1e-4)

    def test_default_matches_point_source(self) -> None:
        # tx_wavefront_radius=0 (the default) must reproduce the existing,
        # already-validated (against Sionna RT) point-source behavior.
        paths = _single_bounce_paths(
            [0.0, 0.0, 1.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 1.0],
            InteractionType.REFLECTION,
        )
        mesh = _ground_plane_mesh("Metal")

        got = compute_received_fields(paths, mesh, 1e9, tx_wavefront_radius=0.0)
        expected = compute_received_fields(paths, mesh, 1e9)

        chex.assert_trees_all_close(got, expected)


class TestTransmissionMatrix:
    def test_transmission_matrix_through_vacuum_is_near_unity(self) -> None:
        # A wave "transmitted" through a vacuum slab should pass essentially
        # unimpeded: |t| ~= 1, since there is no impedance mismatch at all.
        paths = _single_bounce_paths(
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            InteractionType.TRANSMISSION,
        )
        mesh = _ground_plane_mesh("Vacuum")
        frequency = 1e9
        vacuum_with_thickness = Material(
            name="Vacuum", properties=materials["Vacuum"].properties, thickness=0.1
        )
        radio_materials = {"Vacuum": vacuum_with_thickness}

        mat = GeometricFieldSolver().transmission_matrix(
            paths, mesh, frequency, radio_materials
        )

        chex.assert_shape(mat, (1, 1, 2, 2))

        t_s, t_p = refraction_coefficients(1.0, jnp.cos(jnp.array(0.0)))
        chex.assert_trees_all_close(jnp.abs(mat[0, 0, 0, 0]), jnp.abs(t_s), atol=1e-6)
        chex.assert_trees_all_close(jnp.abs(mat[0, 0, 1, 1]), jnp.abs(t_p), atol=1e-6)

    def test_transmission_requires_material_thickness(self) -> None:
        # 'Metal' (like all built-in materials) has no explicit thickness.
        paths = _single_bounce_paths(
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            InteractionType.TRANSMISSION,
        )
        mesh = _ground_plane_mesh("Metal")

        with pytest.raises(Exception, match="finite 'thickness'"):
            compute_received_fields(paths, mesh, 1e9)


class TestDiffractionAgainstSionna:
    """
    Cross-validated against Sionna RT (git commit
    ``a035fc6e239e56e8aab53a07e769e239638caf62``, reports as
    ``sionna-rt==2.0.1``) for a canonical 90-degree wedge corner: two
    30x30 m concrete plates sharing a 30 m edge (matching Sionna's own
    bundled ``scenes/simple_wedge`` geometry), at 3.5 GHz, with TX and RX
    positioned so that the only depth-1 path is edge diffraction around
    the corner.

    Sionna's ``paths.a`` deliberately excludes the carrier propagation
    phase :math:`e^{-j2\\pi f \\tau}` (meant to be combined with delay
    separately via ``Paths.cir()``), whereas DiffeRT's fields already
    include it; the expected values below correct for that before
    comparing. Also, Sionna's ``PlanarArray(polarization="V")`` does not
    correspond to DiffeRT's theta-hat convention for this geometry
    (an orthogonal, benign labeling difference between the two tools'
    spherical-basis conventions) -- ``"H"`` is the matching polarization
    here.
    """

    def _wedge_mesh(self) -> Mesh:
        vertices = jnp.array([
            [0.0, -30.0, -15.0],
            [0.0, -30.0, 15.0],
            [0.0, 0.0, 15.0],
            [0.0, 0.0, -15.0],
            [30.0, 0.0, 15.0],
            [30.0, 0.0, -15.0],
        ])
        triangles = jnp.array([[0, 1, 2], [0, 2, 3], [3, 2, 4], [3, 4, 5]])
        return Mesh(
            vertices=vertices,
            triangles=triangles,
            face_materials=jnp.array([0, 0, 0, 0]),
            material_names=("Concrete",),
        )

    def _fixed_concrete(self) -> Material:
        # Sionna's reference script used a plain (non-frequency-scaled)
        # RadioMaterial with these fixed eps_r/sigma values, rather than
        # DiffeRT's frequency-dependent ITU 'Concrete' table.
        def properties(
            f: Float[ArrayLike, " *#batch"],
        ) -> tuple[Float[Array, " *batch"], Float[Array, " *batch"]]:
            f = jnp.asarray(f, dtype=float)
            return jnp.full_like(f, 5.24), jnp.full_like(f, 0.0462)

        return Material(name="Concrete", properties=properties, thickness=0.1)

    @pytest.mark.parametrize(
        ("tx", "rx", "expected_a"),
        [
            ([-10.0, -5.0, 0.0], [5.0, 10.0, 0.0], -8.981767e-06 + 7.415503e-06j),
            ([-20.0, -5.0, 0.0], [5.0, 20.0, 0.0], -2.5819693e-06 + 2.09219e-06j),
        ],
        ids=["symmetric_near", "symmetric_far"],
    )
    def test_diffraction_matches_sionna(
        self, tx: list[float], rx: list[float], expected_a: complex
    ) -> None:
        mesh = self._wedge_mesh()
        radio_materials = {"Concrete": self._fixed_concrete()}
        frequency = 3.5e9

        # The wedge edge (v2-v3) is triangle 1's (Face A) local edge 2, i.e.,
        # half-edge index 3 * 1 + 2 = 5 (the opposite half-edge, 7, is
        # equally valid and gives the same result).
        vertices = jnp.array([[tx, [0.0, 0.0, 0.0], rx]])
        objects = jnp.array([[-1, 5, -1]])
        mask = jnp.ones((1,), dtype=bool)
        interaction_types = jnp.array([[InteractionType.DIFFRACTION]])
        paths = TracedPaths(
            vertices=vertices,
            objects=objects,
            mask=mask,
            interaction_types=interaction_types,
        )

        a = compute_received_fields(
            paths,
            mesh,
            frequency,
            tx_polarization="H",
            rx_polarization="H",
            radio_materials=radio_materials,
        )

        total_length = 2 * jnp.linalg.norm(jnp.array(tx))
        tau = total_length / 299_792_458.0
        expected = expected_a * jnp.exp(-1j * 2 * jnp.pi * frequency * tau)

        chex.assert_trees_all_close(a[0], expected, rtol=1e-2)


class TestScatteringMatrix:
    """
    Unlike reflection/transmission/diffraction, this is a deterministic
    adaptation of Sionna RT's (inherently Monte Carlo) diffuse scattering
    model -- see :meth:`GeometricFieldSolver.scattering_matrix`'s
    docstring -- so these tests check physically-motivated correctness
    properties rather than exact numerical parity with Sionna RT.
    """

    def _scattering_material(self, s: float, xpd: float = 0.0) -> Material:
        return Material(
            name="Concrete",
            properties=materials["Concrete"].properties,
            scattering_coefficient=s,
            xpd_coefficient=xpd,
        )

    def test_zero_scattering_coefficient_gives_zero_field(self) -> None:
        paths = _single_bounce_paths(
            [0.0, 0.0, 1.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 1.0],
            InteractionType.SCATTERING,
        )
        mesh = _ground_plane_mesh("Concrete")
        radio_materials = {"Concrete": self._scattering_material(s=0.0)}

        mat = GeometricFieldSolver().scattering_matrix(
            paths, mesh, 1e9, radio_materials
        )

        chex.assert_shape(mat, (1, 1, 2, 2))
        assert jnp.iscomplexobj(mat)
        chex.assert_trees_all_close(mat, jnp.zeros_like(mat))

    def test_scattering_matrix_is_finite_and_shaped(self) -> None:
        paths = _single_bounce_paths(
            [0.0, 0.0, 1.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 1.0],
            InteractionType.SCATTERING,
        )
        mesh = _ground_plane_mesh("Concrete")
        radio_materials = {"Concrete": self._scattering_material(s=0.5, xpd=0.2)}

        mat = GeometricFieldSolver().scattering_matrix(
            paths, mesh, 1e9, radio_materials
        )

        chex.assert_shape(mat, (1, 1, 2, 2))
        assert jnp.iscomplexobj(mat)
        assert jnp.all(jnp.isfinite(mat))
        assert jnp.any(mat != 0.0)

    def test_reflection_energy_is_reduced_by_scattering_coefficient(self) -> None:
        # A material with S > 0 should reflect a sqrt(1 - S^2) fraction of
        # the amplitude it would without scattering, conserving energy
        # with the diffuse component.
        paths = _single_bounce_paths(
            [0.0, 0.0, 1.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 1.0],
            InteractionType.REFLECTION,
        )
        mesh = _ground_plane_mesh("Concrete")
        s = 0.6
        radio_materials_s = {"Concrete": self._scattering_material(s=s)}
        radio_materials_0 = {"Concrete": self._scattering_material(s=0.0)}

        mat_s = GeometricFieldSolver().reflection_matrix(
            paths, mesh, 1e9, radio_materials_s
        )
        mat_0 = GeometricFieldSolver().reflection_matrix(
            paths, mesh, 1e9, radio_materials_0
        )

        chex.assert_trees_all_close(mat_s, mat_0 * jnp.sqrt(1.0 - s**2), rtol=1e-5)

    def test_scattering_amplitude_decreases_with_distance(self) -> None:
        # The solid angle subtended by the scattering triangle (as seen
        # from the next vertex) shrinks with distance, so the scattered
        # amplitude should decrease monotonically as the receiver moves
        # further away.
        mesh = _ground_plane_mesh("Concrete")
        radio_materials = {"Concrete": self._scattering_material(s=0.5)}

        magnitudes = []
        for rx_distance in (10.0, 50.0, 200.0):
            paths = _single_bounce_paths(
                [0.0, 0.0, 1.0],
                [5.0, 0.0, 0.0],
                [5.0 + rx_distance, 0.0, 1.0],
                InteractionType.SCATTERING,
            )
            mat = GeometricFieldSolver().scattering_matrix(
                paths, mesh, 1e9, radio_materials
            )
            magnitudes.append(jnp.abs(mat[0, 0, 0, 0]))

        assert magnitudes[0] > magnitudes[1] > magnitudes[2]

    def test_lambertian_pattern_hemisphere_integral_is_one(self) -> None:
        # Sanity-check the Lambertian pattern formula itself (independent
        # of the rest of the pipeline): integrating cos(theta) / pi over
        # the hemisphere (solid angle) should give 1.
        n_theta, n_phi = 400, 400
        theta = jnp.linspace(0.0, jnp.pi / 2, n_theta)
        dtheta = theta[1] - theta[0]
        dphi = 2 * jnp.pi / n_phi
        cos_theta = jnp.cos(theta)
        f_s = cos_theta / jnp.pi
        # d(solid angle) = sin(theta) dtheta dphi
        integral = jnp.sum(f_s * jnp.sin(theta)) * dtheta * dphi * n_phi
        chex.assert_trees_all_close(integral, 1.0, atol=1e-3)
