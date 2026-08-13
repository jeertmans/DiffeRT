import functools
from collections.abc import Mapping
from typing import Any, Literal, TypedDict, Unpack, overload

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Complex, Float

from differt.geometry import Mesh, TracedPaths

from ._constants import c, z_0
from ._material import Material
from ._solvers import AbstractFieldSolver, GeometricFieldSolver


class _GeometricFieldSolverKwargs(TypedDict, total=False):
    tx_polarization: Any
    rx_polarization: Any
    radio_materials: Mapping[str, Material] | None
    tx_wavefront_radii: (
        Float[ArrayLike, "*#batch"]
        | tuple[Float[ArrayLike, "*#batch"], Float[ArrayLike, "*#batch"]]
        | None
    )


@overload
def compute_received_fields(
    paths: TracedPaths,
    mesh: Mesh,
    frequency: Float[ArrayLike, "*#batch"] | None = None,
    *,
    solver: Literal["geometric"] = "geometric",
    **solver_kwargs: Unpack[_GeometricFieldSolverKwargs],
) -> Complex[Array, "*batch"]: ...


@overload
def compute_received_fields(
    paths: TracedPaths,
    mesh: Mesh,
    frequency: Float[ArrayLike, "*#batch"] | None = None,
    *,
    solver: AbstractFieldSolver,
) -> Complex[Array, "*batch"]: ...


def compute_received_fields(
    paths: TracedPaths,
    mesh: Mesh,
    frequency: Float[ArrayLike, "*#batch"] | None = None,
    *,
    solver: AbstractFieldSolver | Literal["geometric"] = "geometric",
    **solver_kwargs: Any,
) -> Complex[Array, "*batch"]:
    """
    Compute the received complex fields for each path.

    This is a convenience wrapper around a :class:`AbstractFieldSolver`,
    defaulting to :class:`GeometricFieldSolver<differt.em.GeometricFieldSolver>`.
    Pass a custom ``solver`` instance to customize how fields are computed,
    e.g., to add support for path interaction types beyond
    :attr:`InteractionType.REFLECTION<differt.em.InteractionType.REFLECTION>`.

    Args:
        paths: The paths.
        mesh: The triangle mesh of the scene.
        frequency: The operating frequency (or frequencies) in Hz.

            May be omitted (left to :data:`None`) when ``solver``'s
            (or the ``tx_polarization`` solver keyword argument's)
            ``tx_polarization`` is an :class:`Antenna<differt.em.Antenna>`
            instance, in which case its own
            :attr:`~differt.em.Antenna.frequency` is used instead.
        solver: The field solver configuration or string shortcut.

            Defaults to a plain :class:`GeometricFieldSolver<differt.em.GeometricFieldSolver>`.
        **solver_kwargs: Parameters passed to the solver configuration when
            it is instantiated from a string shortcut (see
            :class:`GeometricFieldSolver`'s attributes, e.g.,
            ``tx_polarization``, ``rx_polarization``, ``radio_materials``,
            ``tx_wavefront_radii``). Not allowed when ``solver`` is already
            a solver instance.

    Returns:
        The received complex fields of shape ``*batch``.

    Raises:
        ValueError: If ``solver`` is an unknown string shortcut, if
            ``solver_kwargs`` is used together with a solver instance, or
            if ``frequency`` is omitted and cannot be derived from an
            :class:`Antenna<differt.em.Antenna>` instance.
    """
    if isinstance(solver, str):
        if solver == "geometric":
            solver = GeometricFieldSolver(**solver_kwargs)
        else:
            msg = f"Unknown solver: {solver}"
            raise ValueError(msg)
    elif solver_kwargs:
        msg = "solver_kwargs cannot be used when a solver instance is provided."
        raise ValueError(msg)

    if frequency is None:
        tx_polarization = getattr(solver, "tx_polarization", None)
        frequency = getattr(tx_polarization, "frequency", None)
        if frequency is None:
            msg = (
                "'frequency' must be provided explicitly, unless "
                "'tx_polarization' is an 'Antenna' instance."
            )
            raise ValueError(msg)

    return solver.compute_fields(paths, mesh, frequency)


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
