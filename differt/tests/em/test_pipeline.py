import chex
import jax
import jax.numpy as jnp
import pytest

from differt.em import (
    GeometricFieldSolver,
    InteractionType,
    c,
    compute_cir,
    compute_received_fields,
    compute_received_power,
    diffraction_matrix,
    fspl,
    reflection_matrix,
    ris_matrix,
    scattering_matrix,
    transition_matrix,
    transmission_matrix,
)
from differt.geometry import Mesh, TracedPaths


def _single_bounce_paths(interaction_type: int) -> TracedPaths:
    # A single reflection-like bounce off the ground plane (z=0):
    # TX at (0, 0, 1), bounce at (5, 0, 0), RX at (10, 0, 1).
    vertices = jnp.array([[[0.0, 0.0, 1.0], [5.0, 0.0, 0.0], [10.0, 0.0, 1.0]]])
    objects = jnp.array([[-1, 0, -1]])
    mask = jnp.ones(vertices.shape[:-2], dtype=bool)
    interaction_types = jnp.array([[interaction_type]])
    return TracedPaths(
        vertices=vertices,
        objects=objects,
        mask=mask,
        interaction_types=interaction_types,
    )


def _ground_plane_mesh() -> Mesh:
    return Mesh(
        vertices=jnp.array([
            [-100.0, -100.0, 0.0],
            [100.0, -100.0, 0.0],
            [0.0, 100.0, 0.0],
        ]),
        triangles=jnp.array([[0, 1, 2]]),
        face_materials=jnp.array([0]),
        material_names=("Metal",),
    )


def test_los_received_power_matches_fspl() -> None:
    # Set up a direct path of length 10 meters along the x-axis
    # path vertices: (1, 10, 3) where batch is 1 path of length 2
    # Vertices of path: TX at [0, 0, 0], RX at [10, 0, 0]
    vertices = jnp.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    objects = jnp.full(vertices.shape[:-1], -1, dtype=int)
    mask = jnp.ones(vertices.shape[:-2], dtype=bool)
    interaction_types = jnp.empty((*vertices.shape[:-2], 0), dtype=int)
    paths = TracedPaths(
        vertices=vertices,
        objects=objects,
        mask=mask,
        interaction_types=interaction_types,
    )

    frequency = 1e9  # 1 GHz
    mesh = Mesh.empty()  # Empty mesh is fine since order is 0 (no reflections)

    # Compute fields
    fields = compute_received_fields(
        paths,
        mesh,
        frequency,
        tx_polarization="V",
        rx_polarization="V",
    )

    # Compute received power in dBW with z_0=1.0 to compare with FSPL
    power_dbw = compute_received_power(fields, z_0=1.0)
    loss_db = fspl(10.0, frequency, dB=True)

    # Verify that received power is exactly -FSPL
    chex.assert_trees_all_close(power_dbw, -loss_db, atol=1e-5)


def test_compute_cir() -> None:
    # 10m path along x-axis
    vertices = jnp.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    objects = jnp.full(vertices.shape[:-1], -1, dtype=int)
    mask = jnp.ones(vertices.shape[:-2], dtype=bool)
    interaction_types = jnp.empty((*vertices.shape[:-2], 0), dtype=int)
    paths = TracedPaths(
        vertices=vertices,
        objects=objects,
        mask=mask,
        interaction_types=interaction_types,
    )
    frequency = 1e9
    mesh = Mesh.empty()

    fields = compute_received_fields(paths, mesh, frequency)
    delay, fields_out = compute_cir(paths, fields)

    # Expected delay is 10 / c
    expected_delay = 10.0 / c
    chex.assert_trees_all_close(delay, jnp.array([expected_delay]), atol=1e-12)
    chex.assert_trees_all_close(fields_out, fields)


def test_jit_and_gradients() -> None:
    # Verify that received field calculation can be JITted and differentiated
    # We will compute gradients with respect to the receiver position.
    def loss_fn(rx_pos: jax.Array) -> jax.Array:
        tx_pos = jnp.array([0.0, 0.0, 0.0])
        # Vertices shape: (1, 2, 3)
        vertices = jnp.stack([tx_pos, rx_pos])[None, ...]
        objects = jnp.full(vertices.shape[:-1], -1, dtype=int)
        mask = jnp.ones(vertices.shape[:-2], dtype=bool)
        interaction_types = jnp.empty((*vertices.shape[:-2], 0), dtype=int)
        paths = TracedPaths(
            vertices=vertices,
            objects=objects,
            mask=mask,
            interaction_types=interaction_types,
        )
        mesh = Mesh.empty()
        fields = compute_received_fields(paths, mesh, 1e9)
        return jnp.abs(fields[0]) ** 2

    rx_pos_init = jnp.array([10.0, 0.0, 0.0])

    # Test JIT
    jit_loss_fn = jax.jit(loss_fn)
    val = jit_loss_fn(rx_pos_init)
    assert val > 0.0

    # Test Gradients
    grad_loss_fn = jax.jit(jax.grad(loss_fn))
    grads = grad_loss_fn(rx_pos_init)

    # The gradient with respect to x should be negative (power decreases as distance increases)
    assert grads[0] < 0.0
    assert jnp.all(jnp.isfinite(grads))


def test_compute_received_power_coherent_vs_non_coherent() -> None:
    # Set up some dummy fields with 2 paths
    # fields shape: (1, 2)
    fields = jnp.array([[1.0 + 1j, -1.0 + 2j]])
    z_0_val = 50.0

    # Coherent sum:
    # summed_fields = (1.0 + 1j) + (-1.0 + 2j) = 0.0 + 3j
    # power_c = 10 * log10(|3j|^2 / 50) = 10 * log10(9 / 50)
    expected_power_c = 10.0 * jnp.log10(9.0 / z_0_val)
    power_c = compute_received_power(fields, z_0=z_0_val, coherent=True, axis=-1)
    chex.assert_trees_all_close(power_c, jnp.array([expected_power_c]), atol=1e-5)

    # Non-coherent sum:
    # power_nc_1 = |1.0 + 1j|^2 / 50 = 2 / 50
    # power_nc_2 = |-1.0 + 2j|^2 / 50 = 5 / 50
    # total_power = 7 / 50
    # power_nc = 10 * log10(7 / 50)
    expected_power_nc = 10.0 * jnp.log10(7.0 / z_0_val)
    power_nc = compute_received_power(fields, z_0=z_0_val, coherent=False, axis=-1)
    chex.assert_trees_all_close(power_nc, jnp.array([expected_power_nc]), atol=1e-5)


@pytest.mark.require_no_typechecker
def test_compute_received_fields_unknown_solver_string_raises() -> None:
    vertices = jnp.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    objects = jnp.full(vertices.shape[:-1], -1, dtype=int)
    mask = jnp.ones(vertices.shape[:-2], dtype=bool)
    interaction_types = jnp.empty((*vertices.shape[:-2], 0), dtype=int)
    paths = TracedPaths(
        vertices=vertices,
        objects=objects,
        mask=mask,
        interaction_types=interaction_types,
    )
    mesh = Mesh.empty()

    with pytest.raises(ValueError, match="Unknown solver"):
        compute_received_fields(paths, mesh, 1e9, solver="not-a-real-solver")  # type: ignore[arg-type]


def test_compute_received_fields_solver_kwargs_with_instance_raises() -> None:
    vertices = jnp.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    objects = jnp.full(vertices.shape[:-1], -1, dtype=int)
    mask = jnp.ones(vertices.shape[:-2], dtype=bool)
    interaction_types = jnp.empty((*vertices.shape[:-2], 0), dtype=int)
    paths = TracedPaths(
        vertices=vertices,
        objects=objects,
        mask=mask,
        interaction_types=interaction_types,
    )
    mesh = Mesh.empty()
    solver = GeometricFieldSolver()

    with pytest.raises(ValueError, match="solver_kwargs cannot be used"):
        compute_received_fields(paths, mesh, 1e9, solver=solver, tx_polarization="H")


def test_compute_received_fields_missing_frequency_raises() -> None:
    # Default 'tx_polarization' is the plain string "V" (not an
    # AbstractAntenna instance), so it carries no 'frequency' attribute
    # and the frequency cannot be inferred.
    vertices = jnp.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    objects = jnp.full(vertices.shape[:-1], -1, dtype=int)
    mask = jnp.ones(vertices.shape[:-2], dtype=bool)
    interaction_types = jnp.empty((*vertices.shape[:-2], 0), dtype=int)
    paths = TracedPaths(
        vertices=vertices,
        objects=objects,
        mask=mask,
        interaction_types=interaction_types,
    )
    mesh = Mesh.empty()

    with pytest.raises(ValueError, match="'frequency' must be provided explicitly"):
        compute_received_fields(paths, mesh)


@pytest.mark.parametrize(
    ("fn", "match"),
    [
        (transition_matrix, "solver_kwargs cannot be used"),
        (reflection_matrix, "solver_kwargs cannot be used"),
        (diffraction_matrix, "solver_kwargs cannot be used"),
        (scattering_matrix, "solver_kwargs cannot be used"),
        (transmission_matrix, "solver_kwargs cannot be used"),
        (ris_matrix, "solver_kwargs cannot be used"),
    ],
)
def test_matrix_functions_solver_kwargs_with_instance_raises(fn, match) -> None:  # noqa: ANN001
    paths = _single_bounce_paths(InteractionType.REFLECTION)
    mesh = _ground_plane_mesh()
    solver = GeometricFieldSolver()

    with pytest.raises(ValueError, match=match):
        fn(paths, mesh, 1e9, solver=solver, tx_polarization="H")


@pytest.mark.parametrize(
    ("fn", "method_name"),
    [
        (transition_matrix, "transition_matrices"),
        (reflection_matrix, "reflection_matrix"),
        (diffraction_matrix, "diffraction_matrix"),
        (scattering_matrix, "scattering_matrix"),
        (transmission_matrix, "transmission_matrix"),
    ],
)
def test_matrix_functions_default_solver_matches_geometric_field_solver(
    fn,  # noqa: ANN001
    method_name: str,
) -> None:
    # 'solver=None' (the default) must instantiate a plain
    # 'GeometricFieldSolver' from 'solver_kwargs' and delegate to the
    # matching method, exactly as if constructed and called manually.
    paths = _single_bounce_paths(InteractionType.REFLECTION)
    mesh = _ground_plane_mesh()
    frequency = 1e9

    got = fn(paths, mesh, frequency, tx_polarization="H")
    expected = getattr(GeometricFieldSolver(tx_polarization="H"), method_name)(
        paths, mesh, frequency
    )

    chex.assert_trees_all_close(got, expected)


def test_ris_matrix_default_solver_raises_not_implemented() -> None:
    # 'ris_matrix' (the wrapper) must still instantiate a default
    # 'GeometricFieldSolver' when 'solver=None', even though
    # 'GeometricFieldSolver.ris_matrix' unconditionally raises.
    paths = _single_bounce_paths(InteractionType.REFLECTION)
    mesh = _ground_plane_mesh()

    with pytest.raises(NotImplementedError, match="not implemented"):
        ris_matrix(paths, mesh, 1e9)


def test_transition_matrices_is_transition_matrix_alias() -> None:
    from differt.em import transition_matrices

    assert transition_matrices is transition_matrix
