from typing import Any, cast

import chex
import equinox as eqx
import jax.numpy as jnp
import pytest
from jaxtyping import Array, ArrayLike, Complex, Float

from differt.em import (
    AbstractFieldSolver,
    AbstractScatteringPattern,
    BackscatteringPattern,
    Dipole,
    DirectivePattern,
    FarFieldDipoleAntenna,
    GeometricFieldSolver,
    InteractionType,
    LambertianPattern,
    Material,
    compute_received_fields,
    compute_received_power,
    fspl,
    materials,
    refraction_coefficients,
)
from differt.geometry import Mesh, Scene, TracedPaths


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

        mat = GeometricFieldSolver().reflection_matrix(paths, mesh, 1e9)

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
    Validates ``tx_wavefront_radii`` (a non-planar, near-field source),
    which has no Sionna RT equivalent to cross-check against (see
    :class:`GeometricFieldSolver<differt.em.GeometricFieldSolver>`'s
    docstring note). Instead, each test checks a geometric equivalence: a
    point source with ``tx_wavefront_radii=rho0`` at some position must
    give *exactly* the same field as an ideal point source
    (``tx_wavefront_radii=0``) physically moved back by ``rho0`` along
    the direction of the first path segment -- both represent the same
    spherical wavefront arriving at the transmitter.
    """

    def test_los(self) -> None:
        mesh = Mesh.empty()
        frequency = 3.5e9
        rho0 = 2.5
        tx, rx = jnp.array([0.0, 0.0, 0.0]), jnp.array([10.0, 0.0, 0.0])

        paths = _los_paths(tx, rx)
        got = compute_received_fields(paths, mesh, frequency, tx_wavefront_radii=rho0)

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
        got = compute_received_fields(paths, mesh, frequency, tx_wavefront_radii=rho0)

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
            diffraction_paths(tx), wedge, frequency, tx_wavefront_radii=rho0
        )

        k_hat = (diffraction_point - tx) / jnp.linalg.norm(diffraction_point - tx)
        expected = compute_received_fields(
            diffraction_paths(tx - rho0 * k_hat), wedge, frequency
        )

        chex.assert_trees_all_close(got, expected, rtol=1e-4)

    def test_default_matches_point_source(self) -> None:
        # tx_wavefront_radii=0 (the default) must reproduce the existing,
        # already-validated (against Sionna RT) point-source behavior.
        paths = _single_bounce_paths(
            [0.0, 0.0, 1.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 1.0],
            InteractionType.REFLECTION,
        )
        mesh = _ground_plane_mesh("Metal")

        got = compute_received_fields(paths, mesh, 1e9, tx_wavefront_radii=0.0)
        expected = compute_received_fields(paths, mesh, 1e9)

        chex.assert_trees_all_close(got, expected)

    def test_astigmatic_tuple_matches_scalar_when_radii_equal(self) -> None:
        # A '(rho, rho)' tuple must be exactly equivalent to passing 'rho'
        # alone, for any interaction type (here: two reflections).
        paths = _single_bounce_paths(
            [0.0, 0.0, 1.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 1.0],
            InteractionType.REFLECTION,
        )
        mesh = _ground_plane_mesh("Metal")

        got = compute_received_fields(paths, mesh, 1e9, tx_wavefront_radii=(4.0, 4.0))
        expected = compute_received_fields(paths, mesh, 1e9, tx_wavefront_radii=4.0)

        chex.assert_trees_all_close(got, expected)

    def test_astigmatic_go_matches_two_radii_spreading_formula(self) -> None:
        # For a path with no diffraction interaction (here: LoS), the
        # amplitude must follow the general astigmatic ray-tube spreading
        # factor 1/sqrt((rho_s + s)(rho_p + s)) exactly, per the two
        # independent principal radii (see 'spreading_factor').
        mesh = Mesh.empty()
        frequency = 3.5e9
        tx, rx = jnp.array([0.0, 0.0, 0.0]), jnp.array([10.0, 0.0, 0.0])
        rho_s, rho_p = 3.0, 8.0

        paths = _los_paths(tx, rx)
        got = compute_received_fields(
            paths, mesh, frequency, tx_wavefront_radii=(rho_s, rho_p)
        )
        isotropic = compute_received_fields(
            paths, mesh, frequency, tx_wavefront_radii=5.0
        )

        s_tot = 10.0
        expected_ratio = jnp.sqrt(
            ((s_tot + 5.0) ** 2) / ((s_tot + rho_s) * (s_tot + rho_p))
        )

        chex.assert_trees_all_close(
            jnp.abs(got), jnp.abs(isotropic) * expected_ratio, rtol=1e-5
        )

    def test_astigmatic_with_diffraction_raises(self) -> None:
        # Genuinely astigmatic sources (rho_s != rho_p) are not supported
        # together with a DIFFRACTION interaction, since that would
        # require tracking the wavefront's principal-axis orientation
        # along the path, not just its two radii.
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
        paths = TracedPaths(
            vertices=jnp.array([
                [[-10.0, -5.0, 0.0], [0.0, 0.0, 0.0], [5.0, 10.0, 0.0]]
            ]),
            objects=jnp.array([[-1, 5, -1]]),
            mask=jnp.array([True]),
            interaction_types=jnp.array([[InteractionType.DIFFRACTION]]),
        )

        with pytest.raises(Exception, match="astigmatic"):
            compute_received_fields(paths, wedge, 3.5e9, tx_wavefront_radii=(3.0, 8.0))

    def test_planar_go_path_has_no_spreading(self) -> None:
        # A plane wave does not spread at all: the spreading factor must
        # be exactly 1, regardless of the path length, for a path with no
        # DIFFRACTION interaction (here: LoS).
        solver = GeometricFieldSolver(tx_wavefront_radii=None)
        for length in (1.0, 100.0, 1e6):
            paths = _los_paths([0.0, 0.0, 0.0], [length, 0.0, 0.0])
            got = solver.spreading_factor(paths)
            chex.assert_trees_all_close(got, jnp.ones_like(got))

    def test_planar_diffraction_path_has_cylindrical_spreading(self) -> None:
        # A plane wave diffracting off a straight edge produces a
        # cylindrical wave beyond the edge, i.e., a spreading factor of
        # 1/sqrt(s_after), independent of any (infinite) incident
        # distance -- not the naive (and incorrect, vanishing) limit of
        # the point-source formula as the incident distance grows.
        solver = GeometricFieldSolver(tx_wavefront_radii=None)
        s_after = 7.0
        paths = _single_bounce_paths(
            [-100.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, s_after, 0.0],
            InteractionType.DIFFRACTION,
        )
        got = solver.spreading_factor(paths)
        expected = 1.0 / jnp.sqrt(s_after)
        chex.assert_trees_all_close(got[0], expected)

    def test_planar_does_not_raise_with_diffraction(self) -> None:
        # Unlike an astigmatic '(rho_s, rho_p)' tuple, a planar wavefront
        # (None) is supported together with a DIFFRACTION interaction.
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
        paths = TracedPaths(
            vertices=jnp.array([
                [[-10.0, -5.0, 0.0], [0.0, 0.0, 0.0], [5.0, 10.0, 0.0]]
            ]),
            objects=jnp.array([[-1, 5, -1]]),
            mask=jnp.array([True]),
            interaction_types=jnp.array([[InteractionType.DIFFRACTION]]),
        )

        fields = compute_received_fields(paths, wedge, 3.5e9, tx_wavefront_radii=None)
        assert jnp.all(jnp.isfinite(fields))

    def test_far_field_antenna_overrides_tx_wavefront_radii(self) -> None:
        # When 'tx_polarization' is an AbstractAntenna, its own 'wavefront_radii'
        # takes precedence over the explicit 'tx_wavefront_radii'
        # argument -- here, a 'FarFieldDipoleAntenna' always reports a
        # planar wavefront, so passing a (very different) explicit radius
        # must have no effect at all.
        paths = _los_paths([0.0, 0.0, 0.0], [50.0, 0.0, 0.0])
        mesh = Mesh.empty()
        frequency = 1e9
        antenna = FarFieldDipoleAntenna(frequency=frequency)

        got_default = compute_received_fields(
            paths, mesh, frequency, tx_polarization=antenna
        )
        got_explicit = compute_received_fields(
            paths, mesh, frequency, tx_polarization=antenna, tx_wavefront_radii=42.0
        )

        chex.assert_trees_all_close(got_default, got_explicit)

        # And it must indeed differ from a plain (non-far-field) Dipole
        # sharing the same physical parameters, since that one reports a
        # spherical (not planar) wavefront.
        near_field_antenna = Dipole(frequency=frequency)
        got_near_field = compute_received_fields(
            paths, mesh, frequency, tx_polarization=near_field_antenna
        )
        assert not jnp.allclose(got_default, got_near_field)

    def test_dipole_at_tx_position_matches_point_source_spreading(self) -> None:
        # A Dipole centered exactly at the transmitter (the typical usage)
        # reports a wavefront radius of 0 there, i.e., the same spreading
        # behavior as the default point source -- using an AbstractAntenna as
        # 'tx_polarization' must not, by itself, change the spreading law.
        tx = [0.0, 0.0, 0.0]
        rx = [50.0, 0.0, 0.0]
        paths = _los_paths(tx, rx)
        mesh = Mesh.empty()
        frequency = 1e9

        dipole = Dipole(frequency=frequency, center=jnp.array(tx))
        chex.assert_trees_all_close(
            dipole.wavefront_radii(jnp.array(tx)), jnp.array(0.0)
        )

        fields = compute_received_fields(paths, mesh, frequency, tx_polarization=dipole)
        assert jnp.all(jnp.isfinite(fields))


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

        mat = GeometricFieldSolver(radio_materials=radio_materials).transmission_matrix(
            paths, mesh, frequency
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

    .. note::

        The expected values below were regenerated after fixing a sign
        bug in :func:`diffraction_coefficients<differt.em.diffraction_coefficients>`:
        the incident-shadow-boundary terms (:math:`D_1 + D_2`) were
        missing the sign flip relative to the reflection-shadow-boundary
        terms (:math:`D_3`, :math:`D_4`) that Sionna RT's own reference
        implementation applies (``RadioMaterial._diffraction_matrix``
        computes ``d12 = -(d1 + d2)`` before combining with ``d3``/``d4``).
        Without it, the total (LoS + reflection + diffraction) field was
        discontinuous across the ISB -- see
        :class:`TestShadowBoundaryContinuity`, which only ever exercised
        the RSB. A direct re-run against the installed ``sionna-rt==2.0.1``
        confirms the fix is in the right direction and same order of
        magnitude (same sign, within ~10% -- e.g. ``-8.31e-6+8.04e-6j`` for
        ``symmetric_near`` with ``"V"``/``"V"``, matching DiffeRT's own
        ``"V"``/``"V"`` output of ``-8.49e-6+7.11e-6j``, the natural
        polarization correspondence now that the sign bug is fixed);
        the residual gap is a separate, pre-existing approximation
        (:meth:`GeometricFieldSolver.diffraction_matrix
        <differt.em.GeometricFieldSolver.diffraction_matrix>` always
        defaults ``L_r_n``/``L_r_o`` to ``L_i``) outside the scope of
        this fix.
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
            ([-10.0, -5.0, 0.0], [5.0, 10.0, 0.0], 3.186347e-06 - 4.595295e-06j),
            ([-20.0, -5.0, 0.0], [5.0, 20.0, 0.0], 2.364276e-07 - 7.191544e-07j),
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


class TestShadowBoundaryContinuity:
    """
    Regression tests for the continuity of the total (LoS + reflection +
    diffraction) field across the incident and reflection shadow
    boundaries (ISB/RSB), for the canonical 90-degree wedge corner (see
    :class:`TestDiffractionAgainstSionna`). Uniform Theory of Diffraction
    is specifically designed to keep this total field continuous (unlike
    plain geometrical optics, which drops discontinuously to/from zero at
    these boundaries); a previous implementation had a bug (an
    incorrectly clipped material thickness, see
    :meth:`GeometricFieldSolver.diffraction_matrix<differt.em.GeometricFieldSolver.diffraction_matrix>`)
    that broke this continuity specifically for materials with no
    explicit thickness set (i.e., the common case, since
    :data:`materials<differt.em.materials>`'s built-in materials never set
    one).
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

    def test_reflection_shadow_boundary_is_continuous(self) -> None:
        wedge = self._wedge_mesh()
        frequency = 3.5e9
        tx = jnp.array([-10.0, -5.0, 0.0])

        # A window straddling the RSB (found, for this geometry, at ~153.43
        # deg), fine enough to resolve the transition without being so fine
        # that float32 positions on either side of the boundary round to
        # the same value.
        angle = jnp.radians(jnp.linspace(150.0, 158.0, 4000))
        rx = 15.0 * jnp.stack(
            [jnp.cos(angle), jnp.sin(angle), jnp.zeros_like(angle)], axis=-1
        )

        scene = Scene(transmitters=tx, receivers=rx, mesh=wedge)
        los_paths = cast("TracedPaths", scene.trace_paths(order=0))
        refl_paths = cast("TracedPaths", scene.trace_paths(order=1))
        refl_valid = jnp.any(refl_paths.mask, axis=-1)
        (transition_idx,) = jnp.nonzero(jnp.diff(refl_valid.astype(int)))
        assert transition_idx.size > 0, "expected an RSB within the sampled window"
        idx = transition_idx[0]

        los_field = compute_received_fields(los_paths, wedge, frequency)[..., 0]
        refl_field = compute_received_fields(refl_paths, wedge, frequency).sum(axis=-1)
        diffraction_point = jnp.zeros(3)
        diffraction_paths = TracedPaths(
            vertices=jnp.stack(
                [
                    jnp.broadcast_to(tx, rx.shape),
                    jnp.broadcast_to(diffraction_point, rx.shape),
                    rx,
                ],
                axis=-2,
            ),
            objects=jnp.stack(
                [
                    -jnp.ones(rx.shape[0], dtype=int),
                    jnp.full(rx.shape[0], 5, dtype=int),
                    -jnp.ones(rx.shape[0], dtype=int),
                ],
                axis=-1,
            ),
            mask=jnp.ones(rx.shape[0], dtype=bool),
            interaction_types=jnp.full((rx.shape[0], 1), InteractionType.DIFFRACTION),
        )
        diff_field = compute_received_fields(diffraction_paths, wedge, frequency)

        total_power = compute_received_power(los_field + refl_field + diff_field)

        # Right at the boundary, the total field must stay continuous (a
        # small residual step from discretizing a steep-but-continuous
        # transition is expected -- empirically ~0.07 dB at this
        # resolution -- but nowhere near the ~0.4 dB this regression test
        # would have caught before the fix).
        jump = jnp.abs(total_power[idx + 1] - total_power[idx])
        assert jump < 0.15, f"discontinuous jump of {jump:.3f} dB at the RSB"

    def test_incident_shadow_boundary_is_continuous(self) -> None:
        wedge = self._wedge_mesh()
        frequency = 3.5e9
        tx = jnp.array([-10.0, -5.0, 0.0])

        # A window straddling the ISB (found, for this geometry, at ~26.56
        # deg), fine enough to resolve the transition without being so fine
        # that float32 positions on either side of the boundary round to
        # the same value. Regression test for a sign bug in
        # 'diffraction_coefficients' (the D_1+D_2 terms need the opposite
        # sign from D_3/D_4, see 'TestDiffractionAgainstSionna') that
        # previously made this jump ~9.5 dB.
        angle = jnp.radians(jnp.linspace(20.0, 35.0, 4000))
        rx = 15.0 * jnp.stack(
            [jnp.cos(angle), jnp.sin(angle), jnp.zeros_like(angle)], axis=-1
        )

        scene = Scene(transmitters=tx, receivers=rx, mesh=wedge)
        los_paths = cast("TracedPaths", scene.trace_paths(order=0))
        refl_paths = cast("TracedPaths", scene.trace_paths(order=1))
        los_valid = jnp.any(los_paths.mask, axis=-1)
        (transition_idx,) = jnp.nonzero(jnp.diff(los_valid.astype(int)))
        assert transition_idx.size > 0, "expected an ISB within the sampled window"
        idx = transition_idx[0]

        los_field = compute_received_fields(los_paths, wedge, frequency)[..., 0]
        refl_field = compute_received_fields(refl_paths, wedge, frequency).sum(axis=-1)
        diffraction_point = jnp.zeros(3)
        diffraction_paths = TracedPaths(
            vertices=jnp.stack(
                [
                    jnp.broadcast_to(tx, rx.shape),
                    jnp.broadcast_to(diffraction_point, rx.shape),
                    rx,
                ],
                axis=-2,
            ),
            objects=jnp.stack(
                [
                    -jnp.ones(rx.shape[0], dtype=int),
                    jnp.full(rx.shape[0], 5, dtype=int),
                    -jnp.ones(rx.shape[0], dtype=int),
                ],
                axis=-1,
            ),
            mask=jnp.ones(rx.shape[0], dtype=bool),
            interaction_types=jnp.full((rx.shape[0], 1), InteractionType.DIFFRACTION),
        )
        diff_field = compute_received_fields(diffraction_paths, wedge, frequency)

        total_power = compute_received_power(los_field + refl_field + diff_field)

        jump = jnp.abs(total_power[idx + 1] - total_power[idx])
        assert jump < 0.15, f"discontinuous jump of {jump:.3f} dB at the ISB"


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

        mat = GeometricFieldSolver(radio_materials=radio_materials).scattering_matrix(
            paths, mesh, 1e9
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

        mat = GeometricFieldSolver(radio_materials=radio_materials).scattering_matrix(
            paths, mesh, 1e9
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

        mat_s = GeometricFieldSolver(
            radio_materials=radio_materials_s
        ).reflection_matrix(paths, mesh, 1e9)
        mat_0 = GeometricFieldSolver(
            radio_materials=radio_materials_0
        ).reflection_matrix(paths, mesh, 1e9)

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
            mat = GeometricFieldSolver(
                radio_materials=radio_materials
            ).scattering_matrix(paths, mesh, 1e9)
            magnitudes.append(jnp.abs(mat[0, 0, 0, 0]))

        assert magnitudes[0] > magnitudes[1] > magnitudes[2]

    def test_lambertian_pattern_hemisphere_integral_is_one(self) -> None:
        # Test the Lambertian pattern formula itself (independent
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

    def test_default_scattering_pattern_is_lambertian(self) -> None:
        assert isinstance(
            self._scattering_material(s=0.5).scattering_pattern, LambertianPattern
        )

    def test_custom_scattering_pattern_is_used(self) -> None:
        # An isotropic (rather than Lambertian) pattern should give a
        # different amplitude than the default, and should not depend on
        # the scattered direction (unlike the Lambertian one).
        class IsotropicPattern(AbstractScatteringPattern):
            def __call__(
                self,
                k_i: Float[ArrayLike, "*#batch 3"],
                k_s: Float[ArrayLike, "*#batch 3"],
                n: Float[ArrayLike, "*#batch 3"],
            ) -> Float[Array, "*batch"]:
                del k_i, k_s
                return jnp.ones_like(jnp.asarray(n)[..., 0]) / (2.0 * jnp.pi)

        isotropic_pattern = IsotropicPattern()

        paths = _single_bounce_paths(
            [0.0, 0.0, 1.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 1.0],
            InteractionType.SCATTERING,
        )
        mesh = _ground_plane_mesh("Concrete")
        lambertian_material = self._scattering_material(s=0.5)
        isotropic_material = Material(
            name="Concrete",
            properties=lambertian_material.properties,
            scattering_coefficient=0.5,
            scattering_pattern=isotropic_pattern,
        )

        mat_lambertian = GeometricFieldSolver(
            radio_materials={"Concrete": lambertian_material}
        ).scattering_matrix(paths, mesh, 1e9)
        mat_isotropic = GeometricFieldSolver(
            radio_materials={"Concrete": isotropic_material}
        ).scattering_matrix(paths, mesh, 1e9)

        assert jnp.any(mat_lambertian != mat_isotropic)

    def test_directive_pattern_peaks_at_specular_direction(self) -> None:
        k_i = jnp.array([1.0, 0.0, -1.0]) / jnp.sqrt(2.0)
        n = jnp.array([0.0, 0.0, 1.0])
        k_sp = k_i - 2.0 * jnp.sum(k_i * n) * n  # specular direction

        pattern = DirectivePattern(alpha_r=10.0)

        f_s_specular = pattern(k_i, k_sp, n)
        f_s_normal = pattern(k_i, n, n)
        f_s_grazing = pattern(k_i, jnp.array([-1.0, 0.0, 0.0]), n)

        assert f_s_specular > f_s_normal > f_s_grazing

    def test_directive_pattern_hemisphere_integral_approaches_one_for_large_alpha(
        self,
    ) -> None:
        # Unlike the Lambertian pattern, DirectivePattern's normalization
        # is only exact over the full sphere around the specular
        # direction; a large 'alpha_r' concentrates the lobe well inside
        # the physical hemisphere, so little of it is lost below the
        # horizon and the hemispherical integral should approach 1.
        k_i = jnp.array([0.3, 0.0, -jnp.sqrt(1.0 - 0.3**2)])
        n = jnp.array([0.0, 0.0, 1.0])
        pattern = DirectivePattern(alpha_r=50.0)

        n_theta, n_phi = 300, 300
        theta = jnp.linspace(0.0, jnp.pi / 2, n_theta)
        phi = jnp.linspace(0.0, 2 * jnp.pi, n_phi, endpoint=False)
        dtheta = theta[1] - theta[0]
        dphi = phi[1] - phi[0]
        th, ph = jnp.meshgrid(theta, phi, indexing="ij")
        k_s = jnp.stack(
            [jnp.sin(th) * jnp.cos(ph), jnp.sin(th) * jnp.sin(ph), jnp.cos(th)],
            axis=-1,
        )
        f_s = pattern(
            jnp.broadcast_to(k_i, k_s.shape), k_s, jnp.broadcast_to(n, k_s.shape)
        )
        integral = jnp.sum(f_s * jnp.sin(th)) * dtheta * dphi

        chex.assert_trees_all_close(integral, 1.0, atol=1e-2)

    def test_backscattering_pattern_reduces_to_directive_pattern_at_lambda_one(
        self,
    ) -> None:
        k_i = jnp.array([1.0, 0.0, -1.0]) / jnp.sqrt(2.0)
        n = jnp.array([0.0, 0.0, 1.0])
        k_s = jnp.array([0.0, 1.0, 1.0]) / jnp.sqrt(2.0)

        directive = DirectivePattern(alpha_r=4.0)
        backscattering = BackscatteringPattern(alpha_r=4.0, alpha_i=8.0, lambda_=1.0)

        chex.assert_trees_all_close(directive(k_i, k_s, n), backscattering(k_i, k_s, n))

    def test_backscattering_pattern_peaks_at_retroreflection_direction_when_lambda_zero(
        self,
    ) -> None:
        k_i = jnp.array([1.0, 0.0, -1.0]) / jnp.sqrt(2.0)
        n = jnp.array([0.0, 0.0, 1.0])

        pattern = BackscatteringPattern(alpha_r=4.0, alpha_i=10.0, lambda_=0.0)

        f_s_retro = pattern(k_i, -k_i, n)
        f_s_normal = pattern(k_i, n, n)

        assert f_s_retro > f_s_normal

    def test_directive_and_backscattering_patterns_flow_through_scattering_matrix(
        self,
    ) -> None:
        paths = _single_bounce_paths(
            [0.0, 0.0, 1.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 1.0],
            InteractionType.SCATTERING,
        )
        mesh = _ground_plane_mesh("Concrete")
        lambertian_material = self._scattering_material(s=0.5)

        for pattern in (
            DirectivePattern(alpha_r=4.0),
            BackscatteringPattern(alpha_r=4.0, alpha_i=4.0, lambda_=0.3),
        ):
            material = Material(
                name="Concrete",
                properties=lambertian_material.properties,
                scattering_coefficient=0.5,
                scattering_pattern=pattern,
            )
            mat = GeometricFieldSolver(
                radio_materials={"Concrete": material}
            ).scattering_matrix(paths, mesh, 1e9)

            chex.assert_shape(mat, (1, 1, 2, 2))
            assert jnp.iscomplexobj(mat)
            assert jnp.all(jnp.isfinite(mat))
            assert jnp.any(mat != 0.0)
