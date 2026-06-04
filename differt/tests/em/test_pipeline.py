import chex
import jax
import jax.numpy as jnp

from differt.em import (
    c,
    compute_cir,
    compute_received_fields,
    compute_received_power,
    fspl,
)
from differt.geometry import Paths, TriangleMesh


def test_los_received_power_matches_fspl() -> None:
    # Set up a direct path of length 10 meters along the x-axis
    # path vertices: (1, 10, 3) where batch is 1 path of length 2
    # Vertices of path: TX at [0, 0, 0], RX at [10, 0, 0]
    vertices = jnp.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    objects = jnp.full(vertices.shape[:-1], -1, dtype=int)
    paths = Paths(vertices=vertices, objects=objects)

    frequency = 1e9  # 1 GHz
    mesh = TriangleMesh.empty()  # Empty mesh is fine since order is 0 (no reflections)

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
    paths = Paths(vertices=vertices, objects=objects)
    frequency = 1e9
    mesh = TriangleMesh.empty()

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
        paths = Paths(vertices=vertices, objects=objects)
        mesh = TriangleMesh.empty()
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
