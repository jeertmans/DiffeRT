import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, ArrayLike, Float

from differt.em import (
    AbstractRadiationPattern,
    Dipole,
    GeometricFieldSolver,
    InteractionType,
    Material,
    TracedFields,
    c,
    compute_received_fields,
    diffraction_matrix,
    fspl,
    materials,
    reflection_matrix,
    ris_matrix,
    scattering_matrix,
    transition_matrix,
    transmission_matrix,
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


def test_traced_fields_creation_and_properties() -> None:
    fields = jnp.array([[1.0 + 2.0j, 3.0 - 4.0j], [0.0 + 0.0j, 5.0 + 0.0j]])
    delay = jnp.array([[1.0e-8, 2.0e-8], [3.0e-8, 4.0e-8]])
    freq = jnp.array(3.5e9)
    mask = jnp.array([[True, True], [False, True]])

    tf = TracedFields(fields=fields, delay=delay, frequency=freq, mask=mask)

    assert tf.shape == (2, 2)
    assert tf.num_valid_paths == 3

    chex.assert_trees_all_equal(
        tf.masked_fields, jnp.array([1.0 + 2.0j, 3.0 - 4.0j, 5.0 + 0.0j])
    )
    chex.assert_trees_all_equal(tf.masked_delay, jnp.array([1.0e-8, 2.0e-8, 4.0e-8]))


def test_traced_fields_power_and_cir() -> None:
    fields = jnp.array([1.0 + 1.0j, -1.0 + 2.0j])
    delay = jnp.array([1.0e-8, 2.0e-8])
    freq = jnp.array(1.0e9)
    mask = jnp.array([True, True])

    tf = TracedFields(fields=fields, delay=delay, frequency=freq, mask=mask)

    # CIR
    d, f = tf.cir()
    chex.assert_trees_all_equal(d, delay)
    chex.assert_trees_all_equal(f, fields)

    # Power without axis
    power_no_axis = tf.power(z_0=50.0)
    expected_power = 10.0 * jnp.log10(jnp.abs(fields) ** 2 / 50.0)
    chex.assert_trees_all_close(power_no_axis, expected_power, atol=1e-5)

    # Coherent sum along axis 0
    # Sum: (1+1j) + (-1+2j) = 3j -> |3j|^2 = 9
    power_c = tf.power(coherent=True, axis=0, z_0=50.0)
    expected_c = 10.0 * jnp.log10(9.0 / 50.0)
    chex.assert_trees_all_close(power_c, expected_c, atol=1e-5)

    # Non-coherent sum along axis 0
    # |1+1j|^2 = 2, |-1+2j|^2 = 5 -> total = 7
    power_nc = tf.power(coherent=False, axis=0, z_0=50.0)
    expected_nc = 10.0 * jnp.log10(7.0 / 50.0)
    chex.assert_trees_all_close(power_nc, expected_nc, atol=1e-5)


def test_traced_fields_reshape_squeeze_masked_reduce() -> None:
    fields = jnp.array([[[1.0 + 0j], [2.0 + 0j]]])
    delay = jnp.array([[[1.0e-8], [2.0e-8]]])
    freq = jnp.array(1.0e9)
    mask = jnp.array([[[True], [False]]])

    tf = TracedFields(fields=fields, delay=delay, frequency=freq, mask=mask)
    assert tf.shape == (1, 2, 1)

    # Squeeze
    squeezed = tf.squeeze()
    assert squeezed.shape == (2,)

    # Squeeze specific axis
    squeezed_axis = tf.squeeze(axis=0)
    assert squeezed_axis.shape == (2, 1)

    # Reshape
    reshaped = tf.reshape(2)
    assert reshaped.shape == (2,)

    # Masked
    masked_tf = tf.masked()
    assert masked_tf.shape == (1,)
    chex.assert_trees_all_equal(masked_tf.fields, jnp.array([1.0 + 0j]))
    chex.assert_trees_all_equal(masked_tf.delay, jnp.array([1.0e-8]))

    # Reduce
    sum_val = tf.reduce(jnp.abs)
    chex.assert_trees_all_close(sum_val, jnp.array(1.0))


def test_traced_fields_from_paths_and_to_fields() -> None:
    tx = [0.0, 0.0, 0.0]
    rx = [10.0, 0.0, 0.0]
    paths = _los_paths(tx, rx)
    mesh = Mesh.empty()
    frequency = 1.0e9

    # From paths
    tf = TracedFields.from_paths(paths, mesh, frequency)
    assert isinstance(tf, TracedFields)
    chex.assert_trees_all_close(tf.delay[0], jnp.array(10.0 / c))

    expected_power = -fspl(10.0, frequency, dB=True)
    chex.assert_trees_all_close(tf.power(z_0=1.0)[0], expected_power, atol=1e-5)

    # Paths to_fields
    tf2 = paths.to_fields(mesh, frequency)
    chex.assert_trees_all_close(tf.fields, tf2.fields)
    chex.assert_trees_all_close(tf.delay, tf2.delay)


def test_scene_trace_fields() -> None:
    tx = [0.0, 0.0, 0.0]
    rx = [10.0, 0.0, 0.0]
    scene = Scene(
        transmitters=jnp.asarray([tx]),
        receivers=jnp.asarray([rx]),
        mesh=Mesh.empty(),
    )

    tf = scene.trace_fields(order=0, frequency=1.0e9)
    assert isinstance(tf, TracedFields)
    expected_power = -fspl(10.0, 1.0e9, dB=True)
    chex.assert_trees_all_close(tf.power(z_0=1.0)[0, 0, 0], expected_power, atol=1e-5)


def test_ris_interaction_type_and_matrix() -> None:
    assert InteractionType.RIS == 4

    paths = _single_bounce_paths(
        [0.0, 0.0, 1.0],
        [5.0, 0.0, 0.0],
        [10.0, 0.0, 1.0],
        InteractionType.RIS,
    )
    mesh = _ground_plane_mesh("Metal")

    with pytest.raises(NotImplementedError, match="RIS matrix computation"):
        ris_matrix(paths, mesh, 1.0e9)


def test_standalone_matrix_utilities() -> None:
    paths = _single_bounce_paths(
        [0.0, 0.0, 1.0],
        [5.0, 0.0, 0.0],
        [10.0, 0.0, 1.0],
        InteractionType.REFLECTION,
    )
    mesh = _ground_plane_mesh("Metal")
    freq = 1.0e9

    # Transition matrix
    t_mat = transition_matrix(paths, mesh, freq)
    r_mat = reflection_matrix(paths, mesh, freq)

    chex.assert_shape(t_mat, (1, 1, 2, 2))
    chex.assert_trees_all_equal(t_mat, r_mat)

    # Diffraction matrix
    d_mat = diffraction_matrix(paths, mesh, freq)
    chex.assert_shape(d_mat, (1, 1, 2, 2))

    # Scattering matrix
    s_mat = scattering_matrix(paths, mesh, freq)
    chex.assert_shape(s_mat, (1, 1, 2, 2))

    # Transmission matrix with thickness
    mesh_thick = Mesh(
        vertices=mesh.vertices,
        triangles=mesh.triangles,
        face_materials=mesh.face_materials,
        material_names=("Concrete",),
    )
    concrete_thick = Material(
        name="Concrete",
        properties=materials["Concrete"].properties,
        thickness=0.1,
    )
    trans_mat = transmission_matrix(
        paths, mesh_thick, freq, radio_materials={"Concrete": concrete_thick}
    )
    chex.assert_shape(trans_mat, (1, 1, 2, 2))


def test_radiation_pattern_polarization_vectors() -> None:
    # Test that an object implementing polarization_vectors works with compute_received_fields
    class DummyPattern(AbstractRadiationPattern):
        def polarization_vectors(
            self,
            r: Float[ArrayLike, "*#batch 3"],
        ) -> tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]]:
            r_arr = jnp.asarray(r)
            s = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), r_arr.shape)
            p = jnp.broadcast_to(jnp.array([0.0, 1.0, 0.0]), r_arr.shape)
            return s, p

    pattern = DummyPattern(frequency=1.0e9)
    paths = _los_paths([0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
    mesh = Mesh.empty()

    fields = compute_received_fields(
        paths, mesh, tx_polarization=pattern, rx_polarization=pattern
    )
    assert jnp.all(jnp.isfinite(fields))


def test_traced_fields_float_mask_properties() -> None:
    fields = jnp.array([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j])
    delay = jnp.array([1.0e-8, 2.0e-8, 3.0e-8])
    freq = jnp.array(1.0e9)
    mask = jnp.array([0.9, 0.4, 0.6])

    tf = TracedFields(
        fields=fields,
        delay=delay,
        frequency=freq,
        mask=mask,
        confidence_threshold=0.5,
    )

    assert tf.num_valid_paths == 2

    chex.assert_trees_all_equal(tf.masked_fields, jnp.array([1.0 + 1.0j, 3.0 + 3.0j]))
    chex.assert_trees_all_equal(tf.masked_delay, jnp.array([1.0e-8, 3.0e-8]))


def test_traced_fields_squeeze_errors() -> None:
    fields_0d = jnp.array(1.0 + 0j)
    delay_0d = jnp.array(1.0e-8)
    freq = jnp.array(1.0e9)
    mask_0d = jnp.array(True)

    tf_0d = TracedFields(fields=fields_0d, delay=delay_0d, frequency=freq, mask=mask_0d)

    with pytest.raises(ValueError, match="Cannot squeeze a 0-dimensional batch"):
        tf_0d.squeeze(axis=0)

    fields_1d = jnp.array([1.0 + 0j, 2.0 + 0j])
    delay_1d = jnp.array([1.0e-8, 2.0e-8])
    mask_1d = jnp.array([True, True])

    tf_1d = TracedFields(fields=fields_1d, delay=delay_1d, frequency=freq, mask=mask_1d)

    with pytest.raises(ValueError, match="out-of-bounds"):
        tf_1d.squeeze(axis=5)


def test_traced_fields_reduce_with_float_mask() -> None:
    fields = jnp.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])
    delay = jnp.array([1.0e-8, 2.0e-8, 3.0e-8])
    freq = jnp.array(1.0e9)
    mask = jnp.array([0.9, 0.4, 0.6])

    tf = TracedFields(
        fields=fields,
        delay=delay,
        frequency=freq,
        mask=mask,
        confidence_threshold=0.5,
    )

    sum_val = tf.reduce(jnp.abs)
    expected = jnp.sum(jnp.abs(fields) * mask)
    chex.assert_trees_all_close(sum_val, expected)


@pytest.mark.require_no_typechecker
def test_traced_fields_from_paths_unknown_solver() -> None:
    # The public signature types 'solver' as 'AbstractFieldSolver | Literal["geometric"]',
    # so under the project's runtime type checking (jaxtyping + beartype), passing any
    # other string is rejected by beartype itself before 'from_paths' body ever runs.
    # This test only runs in the CI job with type checking disabled, matching the
    # 'Unknown solver' precedent in 'test_scene.py'.
    paths = _los_paths([0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
    mesh = Mesh.empty()

    with pytest.raises(ValueError, match="Unknown solver"):
        TracedFields.from_paths(paths, mesh, 1.0e9, solver="not-a-solver")


def test_traced_fields_from_paths_solver_kwargs_with_instance() -> None:
    paths = _los_paths([0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
    mesh = Mesh.empty()
    solver = GeometricFieldSolver()

    with pytest.raises(
        ValueError,
        match="solver_kwargs cannot be used when a solver instance is provided",
    ):
        TracedFields.from_paths(paths, mesh, 1.0e9, solver=solver, tx_polarization="H")


def test_traced_fields_from_paths_frequency_inferred_from_antenna() -> None:
    paths = _los_paths([0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
    mesh = Mesh.empty()
    antenna = Dipole(frequency=2.4e9)
    solver = GeometricFieldSolver(tx_polarization=antenna)

    tf = TracedFields.from_paths(paths, mesh, solver=solver)

    chex.assert_trees_all_close(tf.frequency, jnp.array(2.4e9))


def test_traced_fields_from_paths_missing_frequency() -> None:
    paths = _los_paths([0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
    mesh = Mesh.empty()
    solver = GeometricFieldSolver()  # tx_polarization defaults to "V", no .frequency

    with pytest.raises(ValueError, match="'frequency' must be provided explicitly"):
        TracedFields.from_paths(paths, mesh, solver=solver)


def test_jit_and_gradients_on_traced_fields() -> None:
    def loss_fn(rx_pos: jax.Array) -> jax.Array:
        tx_pos = jnp.array([0.0, 0.0, 0.0])
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
        tf = paths.to_fields(mesh, 1.0e9)
        return jnp.abs(tf.fields[0]) ** 2

    rx_pos_init = jnp.array([10.0, 0.0, 0.0])

    jit_loss = jax.jit(loss_fn)
    val = jit_loss(rx_pos_init)
    assert val > 0.0

    grad_loss = jax.jit(jax.grad(loss_fn))
    grads = grad_loss(rx_pos_init)
    assert grads[0] < 0.0
    assert jnp.all(jnp.isfinite(grads))
