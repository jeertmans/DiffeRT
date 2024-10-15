"""
Common antenna utilities.

All utilities currently assume propagation in vacuum.
"""

from abc import abstractmethod
from dataclasses import KW_ONLY
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, ArrayLike, Float, Inexact, jaxtyped

from differt.geometry.utils import normalize
from differt.plotting import PlotOutput, draw_surface

from .constants import c, epsilon_0, mu_0


@jaxtyped(typechecker=typechecker)
class Antenna(eqx.Module):
    """An antenna class, must be subclassed."""

    frequency: Float[Array, " "] = eqx.field(converter=jnp.asarray)
    """The frequency :math:`f` at which the antenna is operating."""
    _: KW_ONLY
    center: Float[Array, "3"] = eqx.field(
        converter=jnp.asarray, default_factory=lambda: jnp.array([0.0, 0.0, 0.0])
    )
    """The center position of the antenna, from which the fields are radiated.

    Default value is the origin.
    """

    @property
    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def period(self) -> Float[Array, " "]:
        """The period :math:`T = 1/f`."""
        return 1 / self.frequency

    @property
    @jaxtyped(typechecker=typechecker)
    def angular_frequency(self) -> Float[Array, " "]:
        r"""The angular frequency :math:`\omega = 2 \pi f`."""
        return 2 * jnp.pi * self.frequency

    @property
    @jaxtyped(typechecker=typechecker)
    def wavelength(self) -> Float[Array, " "]:
        r"""The wavelength :math:`\lambda = c / f`."""
        return c * self.period

    @property
    @jaxtyped(typechecker=typechecker)
    def wavenumber(self) -> Float[Array, " "]:
        r"""The wavenumber :math:`k = \omega / c`."""
        return self.angular_frequency / c

    @abstractmethod
    def fields(
        self, r: Float[Array, "*#batch 3"], t: Float[Array, "*#batch 3"] | None = None
    ) -> tuple[Inexact[Array, "*batch 3"], Inexact[Array, "*batch 3"]]:
        r"""
        Compute electrical and magnetical fields in vacuum at given position and (optional) time.

        Args:
            r: The array of positions.
            t: The array of time instants.

                If not provided, initial time instant
                is assumed.

        Returns:
            The electrical :math:`\vec{E}` and magnetical :math:`\vec{B}` fields.

            Fields can be either real of complex-valued.
        """

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def pointing_vector(
        self, r: Float[Array, "*#batch 3"], t: Float[Array, "*#batch 3"] | None = None
    ) -> Inexact[Array, "*batch 3"]:
        r"""
        Compute the pointing vector in vacuum at given position and (optional) time.

        Args:
            r: The array of positions.
            t: The array of time instants.

                If not provided, initial time instant
                is assumed.

        Returns:
            The pointing vector :math:`\vec{S}`.

            It can be either real of complex-valued.
        """
        e, b = self.fields(r, t)

        h = b / mu_0

        return jnp.cross(e, h)

    def plot_radiation_pattern(
        self,
        num_points: int = int(1e2),
        distance: Float[ArrayLike, " "] = 1.0,
        num_wavelengths: Float[ArrayLike, " "] | None = None,
        **kwargs: Any,
    ) -> PlotOutput:
        """
        Plot the radiation pattern (normalized power) of this antenna.

        The power is computed on points on an sphere around the antenna.

        Args:
            num_points: The number of points to sample along the elevation axis.

                Twice this number of points are sampled on the aximutal axis.
            distance: The distance from the antenna at which power samples
                are evaluated.
            num_wavelengths: If provided, supersedes ``distance`` by setting
                the distance relatively to the :attr:`wavelength`.
            kwargs: Keyword arguments passed to
                :func:`draw_surface<differt.plotting.draw_surface>`.
        """
        if num_wavelengths is not None:
            distance = jnp.asarray(num_wavelengths) * self.wavelength
        else:
            distance = jnp.asarray(distance)

        u = jnp.linspace(0, 2 * jnp.pi, num_points * 2)
        v = jnp.linspace(0, jnp.pi, num_points)
        x = jnp.outer(jnp.cos(u), jnp.sin(v))
        y = jnp.outer(jnp.sin(u), jnp.sin(v))
        z = jnp.outer(jnp.ones_like(u), jnp.cos(v))

        r = self.center + distance * jnp.stack((x, y, z), axis=-1)

        s = self.pointing_vector(r)

        p = jnp.linalg.norm(s, axis=-1, keepdims=True)

        gain = p / p.max()

        r *= gain
        gain = jnp.squeeze(gain, axis=-1)

        return draw_surface(
            x=r[..., 0], y=r[..., 1], z=r[..., 2], colors=gain, **kwargs
        )


@jaxtyped(typechecker=typechecker)
class Dipole(Antenna):
    r"""
    A simple electrical dipole.

    Args:
        frequency: The frequency at which the antenna is operating.
        num_wavelengths: The length of the dipole, relative to the wavelength.
        length: The absolute length of the dipole, supersedes ``num_wavelengths``.
        moment: The dipole moment.
        current: The current (in A) flowing in the dipole.

            If this is provided, which is the default, the only the direction of the moment
            vector is used, and its insensity is set to match the dipole moment with
            specified current.
        charge: The dipole charge (in Coulomb), assuming opposite charges on either ends of the dipole.

            If this is provided, this takes precedence over ``current``.
        center: The center position of the antenna, from which the fields are radiated.

    Examples:
        The following example shows how to plot the radiation
        pattern (antenna power) at 1 meter.

        .. plotly::
            :fig-vars: fig

            >>> from differt.em.antenna import Dipole
            >>>
            >>> ant = Dipole(frequency=1e9)
            >>> fig = ant.plot_radiation_pattern(backend="plotly")
            >>> fig  # doctest: +SKIP

        The second example shows how to plot the radiation
        pattern (antenna power) at 1 meter, but only
        in the x-z plane, for multiple dipole lengths.

        .. plot::

            >>> from differt.em.antenna import Dipole
            >>>
            >>> theta = jnp.linspace(0, 2 * jnp.pi)
            >>> r = jnp.stack(
            ...     (jnp.cos(theta), jnp.zeros_like(theta), jnp.sin(theta)), axis=-1
            ... )
            >>> for ratio in [0.25, 0.5, 0.75, 1.0, 2.0]:
            >>>     ant = Dipole(1e9, ratio)
            >>>     power = jnp.linalg.norm(ant.pointing_vector(r), axis=-1)
            >>>     plt.polar(theta, power, label=fr"$\ell/\lambda = {ratio:1.2f}$")

            >>> plt.gca().set_rscale("log")
            >>> plt.legend()
            >>> plt.show()
    """

    length: Float[Array, " "]
    """Dipole length (in meter)."""
    moment: Float[Array, "3"]
    """Dipole moment (in Coulomb-meter)."""

    @jaxtyped(typechecker=typechecker)
    def __init__(
        self,
        frequency: Float[ArrayLike, " "],
        num_wavelengths: Float[ArrayLike, " "] = 0.5,
        *,
        length: Float[ArrayLike, " "] | None = None,
        moment: Float[ArrayLike, " "] | None = jnp.array([0.0, 0.0, 1.0]),
        current: Float[ArrayLike, " "] | None = 1.0,
        charge: Float[ArrayLike, " "] | None = None,
        center: Float[Array, "3"] = jnp.array([0.0, 0.0, 0.0]),
    ) -> None:
        super().__init__(jnp.asarray(frequency), center=center)

        if length is not None:
            self.length = jnp.asarray(length)
        else:
            self.length = jnp.asarray(num_wavelengths) * self.wavelength

        moment = jnp.array(moment)

        if charge is not None:
            moment *= jnp.asarray(charge) * self.length / jnp.linalg.norm(moment)
        elif current is not None:
            moment *= (
                jnp.asarray(current)
                * self.length
                / (jnp.linalg.norm(moment) * self.angular_frequency)
            )

        self.moment = moment  # type: ignore[reportAttributeAccessIssue]

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def fields(
        self, r: Float[Array, "*#batch 3"], t: Float[Array, "*#batch 3"] | None = None
    ) -> tuple[Inexact[Array, "*batch 3"], Inexact[Array, "*batch 3"]]:
        r_hat, r = normalize(r - self.center, keepdims=True)
        p = self.moment
        w = self.angular_frequency
        k = self.wavenumber
        k_k = k * k
        r_inv = 1 / r
        j_k_r = 1j * k * r

        factor = 1 / (4 * jnp.pi * epsilon_0)

        r_x_p = jnp.cross(r_hat, p)
        r_d_p = jnp.sum(r_hat * p, axis=-1, keepdims=True)

        e = (
            factor
            * (
                k_k * jnp.cross(r_x_p, r_hat)
                + r_inv * r_inv * (r_inv - 1j * k) * (3 * r_hat * r_d_p - p)
            )
            * r_inv
        )
        b = (factor * k_k / c) * r_x_p * (1 - 1 / j_k_r) * r_inv

        exp = jnp.exp(j_k_r - 1j * w * t) if t else jnp.exp(j_k_r)

        e *= exp
        b *= exp

        return e, b
