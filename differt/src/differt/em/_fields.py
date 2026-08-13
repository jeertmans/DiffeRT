import typing
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Complex, Float, Int, Num

from differt.geometry._mesh import Mesh
from differt.geometry._paths import TracedPaths

from ._constants import c, z_0
from ._solvers import AbstractFieldSolver, GeometricFieldSolver

if TYPE_CHECKING or hasattr(typing, "GENERATING_DOCS"):
    from typing import Self
else:
    Self = Any  # Because runtime type checking from 'beartype' will fail when combined with 'jaxtyping'


@jax.jit(static_argnames=("coherent", "axis"))
def _compute_power(
    fields: Complex[Array, "*batch"],
    z_0: Float[ArrayLike, ""] | float = z_0,
    coherent: bool = True,
    axis: int | None = None,
) -> Float[Array, "..."]:
    if axis is not None:
        if coherent:
            summed_fields = jnp.sum(fields, axis=axis)
            power = jnp.abs(summed_fields) ** 2 / z_0
        else:
            power = jnp.sum(jnp.abs(fields) ** 2 / z_0, axis=axis)
    else:
        power = jnp.abs(fields) ** 2 / z_0
    return 10.0 * jnp.log10(power)


class TracedFields(eqx.Module):
    """
    A convenient wrapper class around received electromagnetic fields, propagation delays, and masks.

    This class represents the EM fields computed for a set of paths (e.g., as produced by
    :class:`~differt.geometry.TracedPaths`).
    """

    fields: Complex[Array, "*batch"]
    """The complex received electric fields (or channel gains) for each path."""
    delay: Float[Array, "*batch"]
    """The propagation delay (in seconds) for each path."""
    frequency: Float[Array, "*#batch"]
    """The operating frequency (in Hz)."""
    mask: Bool[Array, " *batch"] | Float[Array, " *batch"]
    """A mask to indicate which paths are valid and contribute to the received field.

    The mask is kept separately to :attr:`fields` so that we can keep information about
    batch ``*batch`` dimensions, which would not be possible if we were to directly
    store valid paths.

    If :attr:`mask` contains floating-point values, then they are interpreted as confidence
    values between 0 and 1, where values greater than or equal to :attr:`confidence_threshold`
    are considered valid.
    """
    confidence_threshold: Float[ArrayLike, " "] = 0.5
    """A threshold used to decide whether a given path is valid or not.

    A path is considered valid if its confidence is greater than or equal to this threshold.
    Unused if :attr:`mask` is of type :class:`bool`.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """The batch shape of the fields."""
        return self.fields.shape

    @property
    def num_valid_paths(self) -> Int[Array, ""]:
        """The number of paths kept by :attr:`mask`.

        The output value can be traced by JAX.
        """
        if self.mask.dtype == jnp.bool_:
            return self.mask.sum()
        return (self.mask >= self.confidence_threshold).sum()

    @property
    def masked_fields(self) -> Complex[Array, "num_valid_paths"]:
        """The array of masked fields, with batched dimensions flattened into one."""
        fields = self.fields.reshape(-1)
        mask = self.mask.reshape(-1)
        if mask.dtype != jnp.bool_:
            mask = mask >= self.confidence_threshold
        return fields[mask]

    @property
    def masked_delay(self) -> Float[Array, "num_valid_paths"]:
        """The array of masked propagation delays, with batched dimensions flattened into one."""
        delay = self.delay.reshape(-1)
        mask = self.mask.reshape(-1)
        if mask.dtype != jnp.bool_:
            mask = mask >= self.confidence_threshold
        return delay[mask]

    def reshape(self, *batch: int) -> Self:
        """
        Return a new fields instance with reshaped batch dimensions to match a given shape.

        Args:
            batch: New batch shape.

        Returns:
            A new fields instance with specified batch dimensions.
        """
        fields = self.fields.reshape(*batch)
        resolved_batch = fields.shape
        delay = self.delay.reshape(*resolved_batch)
        mask = self.mask.reshape(*resolved_batch)

        return eqx.tree_at(
            lambda f: (f.fields, f.delay, f.mask),
            self,
            (fields, delay, mask),
        )

    def squeeze(self, axis: int | Sequence[int] | None = None) -> Self:
        """
        Return a new fields instance by squeezing one or more axes of batch dimensions.

        Args:
            axis: See :func:`jax.numpy.squeeze` for allowed values.

        Returns:
            A new fields instance with squeezed batch dimensions.

        Raises:
            ValueError: If one of the provided axes is out-of-bounds,
                or if trying to squeeze a 0-dimensional batch.
        """
        ndim = self.fields.ndim
        if axis is not None and ndim == 0:
            msg = "Cannot squeeze a 0-dimensional batch!"
            raise ValueError(msg)
        if isinstance(axis, int):
            axis = (axis,)
        if isinstance(axis, Sequence):
            axis = tuple(a + ndim if a < 0 else a for a in axis)

            if any(ax >= ndim or ax < 0 for ax in axis):
                msg = "One of the provided axes is out-of-bounds!"
                raise ValueError(msg)

        fields = self.fields.squeeze(axis)
        delay = self.delay.squeeze(axis)
        mask = self.mask.squeeze(axis)

        return eqx.tree_at(
            lambda f: (f.fields, f.delay, f.mask),
            self,
            (fields, delay, mask),
        )

    def masked(self) -> Self:
        """
        Return a flattened version of this object that only keeps valid paths.

        The returned object has all batch dimensions flattened into one,
        keeping only the paths where :attr:`mask` is :data:`True` (or where
        :attr:`mask` is greater than or equal to :attr:`confidence_threshold`).

        Returns:
            A new fields instance with flattened batch dimensions and only valid paths.
        """
        fields_flat = self.fields.reshape(-1)
        delay_flat = self.delay.reshape(-1)
        mask = self.mask.reshape(-1)

        valid = mask >= self.confidence_threshold if mask.dtype != jnp.bool_ else mask

        return eqx.tree_at(
            lambda f: (f.fields, f.delay, f.mask),
            self,
            (
                fields_flat[valid],
                delay_flat[valid],
                mask[valid],
            ),
        )

    def power(
        self,
        coherent: bool = True,
        axis: int | None = None,
        z_0: Float[ArrayLike, ""] | float = z_0,
    ) -> Float[Array, "..."]:
        """
        Compute the received power from the received fields (in dBW).

        Args:
            coherent: Whether to sum coherently (vector sum of fields before power)
                or non-coherently (power sum of individual fields).
                Only active if ``axis`` is not None.
            axis: The axis along which to sum the fields. If None, no sum is performed.
            z_0: The reference impedance.

        Returns:
            The received power in dBW.
        """
        return _compute_power(self.fields, z_0=z_0, coherent=coherent, axis=axis)

    def cir(self) -> tuple[Float[Array, "*batch"], Complex[Array, "*batch"]]:
        """
        Return the Channel Impulse Response (CIR) as (delay, fields) pairs.

        Returns:
            A tuple of (delay, fields) where delay and fields have the same shape.
        """
        return self.delay, self.fields

    def reduce(
        self,
        fun: Callable[[Complex[Array, "*batch"]], Num[Array, " *batch"]],
        axis: int | Sequence[int] | None = None,
    ) -> Num[Array, ""] | Num[Array, " *reduced_batch"]:
        """
        Apply a function on all fields and accumulate the result into a scalar value (or an array if ``axis`` is provided).

        Args:
            fun: Function to apply on fields.
            axis: See :func:`jax.numpy.sum` for allowed values.

        Returns:
            The sum of the results, with contributions from
            invalid paths that are set to zero.
        """
        if self.mask.dtype != jnp.bool_:
            return jnp.sum(fun(self.fields) * self.mask, axis=axis)

        return jnp.sum(fun(self.fields), axis=axis, where=self.mask)

    @classmethod
    def from_paths(
        cls,
        paths: TracedPaths,
        mesh: Mesh,
        frequency: Float[ArrayLike, "*#batch"] | None = None,
        *,
        solver: AbstractFieldSolver | Literal["geometric"] = "geometric",
        **solver_kwargs: Any,
    ) -> Self:
        """
        Compute the received electromagnetic fields for each path and return a :class:`TracedFields` instance.

        Args:
            paths: The traced paths.
            mesh: The triangle mesh of the scene.
            frequency: The operating frequency (or frequencies) in Hz.

                May be omitted (left to :data:`None`) when ``solver``'s
                ``tx_polarization`` is an :class:`Antenna<differt.em.Antenna>`
                instance, in which case its own
                :attr:`~differt.em.Antenna.frequency` is used instead.
            solver: The field solver configuration or string shortcut.

                Defaults to :class:`GeometricFieldSolver<differt.em.GeometricFieldSolver>`.
            **solver_kwargs: Parameters passed to the solver configuration when
                it is instantiated from a string shortcut (e.g.,
                ``tx_polarization``, ``rx_polarization``, ``radio_materials``,
                ``tx_wavefront_radii``).

        Returns:
            A :class:`TracedFields` instance wrapping the computed fields, propagation
            delays, operating frequency, and validity mask.

        Raises:
            ValueError: If an unknown solver name is passed, if ``solver_kwargs``
                are supplied alongside a solver instance, or if ``frequency``
                is omitted and cannot be inferred from the transmitter antenna.
        """
        if isinstance(solver, str):
            if solver == "geometric":
                solver_instance: AbstractFieldSolver = GeometricFieldSolver(
                    **solver_kwargs
                )
            else:
                msg = f"Unknown solver: {solver}"
                raise ValueError(msg)
        elif solver_kwargs:
            msg = "solver_kwargs cannot be used when a solver instance is provided."
            raise ValueError(msg)
        else:
            solver_instance = solver

        if frequency is None:
            tx_polarization = getattr(solver_instance, "tx_polarization", None)
            frequency_val = getattr(tx_polarization, "frequency", None)
            if frequency_val is None:
                msg = (
                    "'frequency' must be provided explicitly, unless "
                    "'tx_polarization' is an 'Antenna' instance."
                )
                raise ValueError(msg)
        else:
            frequency_val = frequency

        frequency_arr = jnp.asarray(frequency_val)
        fields = solver_instance.compute_fields(paths, mesh, frequency_arr)

        path_segments = jnp.diff(paths.vertices, axis=-2)
        lengths = jnp.linalg.norm(path_segments, axis=-1).sum(axis=-1)
        delay = lengths / c

        return cls(
            fields=fields,
            delay=delay,
            frequency=frequency_arr,
            mask=paths.mask,
            confidence_threshold=paths.confidence_threshold,
        )
