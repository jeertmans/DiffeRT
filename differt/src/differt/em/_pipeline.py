import functools
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Complex, Float

from differt.geometry import Mesh, TracedPaths

from ._constants import c, z_0
from ._material import Material
from ._solvers import AbstractFieldSolver, GeometricFieldSolver


def compute_received_fields(
    paths: TracedPaths,
    mesh: Mesh,
    frequency: Float[ArrayLike, "*#batch"],
    tx_polarization: Any = "V",
    rx_polarization: Any = "V",
    radio_materials: Mapping[str, Material] | None = None,
    *,
    solver: AbstractFieldSolver | None = None,
    tx_wavefront_radius: Float[ArrayLike, "*#batch"] = 0.0,
) -> Complex[Array, "*batch"]:
    """
    Compute the received complex fields for each path.

    This is a convenience wrapper around a :class:`AbstractFieldSolver`,
    defaulting to :class:`GeometricFieldSolver<differt.em.GeometricFieldSolver>`.
    Pass a custom ``solver`` to customize how fields are computed, e.g., to
    add support for path interaction types beyond
    :attr:`InteractionType.REFLECTION<differt.em.InteractionType.REFLECTION>`.

    Args:
        paths: The paths.
        mesh: The triangle mesh of the scene.
        frequency: The operating frequency (or frequencies) in Hz.
        tx_polarization: The transmitter antenna polarization or pattern.
        rx_polarization: The receiver antenna polarization or pattern.
        radio_materials: The dictionary of material properties.
        solver: The field solver to use.

            Defaults to a plain :class:`GeometricFieldSolver<differt.em.GeometricFieldSolver>`.
        tx_wavefront_radius: The radius of curvature of the incident
            wavefront at the transmitter, for a non-planar (near-field)
            source (e.g., a focused beam); ``0`` (the default) is an ideal
            point source, matching Sionna RT's implicit assumption. See
            :meth:`GeometricFieldSolver.compute_fields<differt.em.GeometricFieldSolver.compute_fields>`.

    Returns:
        The received complex fields of shape ``*batch``.
    """
    if solver is None:
        solver = GeometricFieldSolver()

    return solver.compute_fields(
        paths,
        mesh,
        frequency,
        tx_polarization=tx_polarization,
        rx_polarization=rx_polarization,
        radio_materials=radio_materials,
        tx_wavefront_radius=tx_wavefront_radius,
    )


@functools.partial(jax.jit, static_argnames=("coherent", "axis"))
def compute_received_power(
    fields: Complex[Array, "*batch"],
    z_0: Float[ArrayLike, ""] | float = z_0,
    coherent: bool = True,
    axis: int | None = None,
) -> Float[Array, "..."]:
    """
    Compute the received power from the received fields (in dBW).

    Args:
        fields: The complex received fields.
        z_0: The reference impedance.
        coherent: Whether to sum coherently (vector sum of fields before power)
            or non-coherently (power sum of individual fields).
            Only active if ``axis`` is not None.
        axis: The axis along which to sum the fields. If None, no sum is performed.

    Returns:
        The received power in dBW.
    """
    if axis is not None:
        if coherent:
            summed_fields = jnp.sum(fields, axis=axis)
            power = jnp.abs(summed_fields) ** 2 / z_0
        else:
            power = jnp.sum(jnp.abs(fields) ** 2 / z_0, axis=axis)
    else:
        power = jnp.abs(fields) ** 2 / z_0
    return 10.0 * jnp.log10(power)


def compute_cir(
    paths: TracedPaths,
    fields: Complex[Array, "*batch"],
) -> tuple[Float[Array, "*batch"], Complex[Array, "*batch"]]:
    """
    Compute the Channel Impulse Response (CIR) as (delay, fields) pairs.

    Args:
        paths: The paths.
        fields: The complex received fields.

    Returns:
        A tuple of (delay, fields) where delay and fields have the same shape.
    """
    path_segments = jnp.diff(paths.vertices, axis=-2)
    lengths = jnp.linalg.norm(path_segments, axis=-1).sum(axis=-1)
    delay = lengths / c
    return delay, fields
