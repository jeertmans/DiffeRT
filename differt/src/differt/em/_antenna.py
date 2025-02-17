from abc import abstractmethod
from dataclasses import KW_ONLY
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Inexact

from differt.geometry import normalize
from differt.plotting import PlotOutput, draw_surface
from differt.utils import dot, safe_divide

from ._constants import c, epsilon_0, mu_0


@jax.jit
def pointing_vector(
    e: Inexact[ArrayLike, "*#batch 3"],
    b: Inexact[ArrayLike, "*#batch 3"],
) -> Inexact[Array, "*batch 3"]:
    r"""
    Compute the pointing vector in vacuum at from electric :math:`\vec{E}` and magnetic :math:`\vec{B}` fields.

    Args:
        e: The electrical field.
        b: The magnetical field.

    Returns:
        The pointing vector :math:`\vec{S}`.

        It can be either real of complex-valued.
    """
    h = jnp.asarray(b) / mu_0

    return jnp.cross(jnp.asarray(e), h)


class BaseAntenna(eqx.Module):
    """An antenna class, base class for :class:`Antenna` and :class:`RadiationPattern`."""

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
    def period(self) -> Float[Array, " "]:
        """The period :math:`T = 1/f`."""
        return 1 / self.frequency

    @property
    def angular_frequency(self) -> Float[Array, " "]:
        r"""The angular frequency :math:`\omega = 2 \pi f`."""
        return 2 * jnp.pi * self.frequency

    @property
    def wavelength(self) -> Float[Array, " "]:
        r"""The wavelength :math:`\lambda = c / f`."""
        return c * self.period

    @property
    def wavenumber(self) -> Float[Array, " "]:
        r"""The wavenumber :math:`k = \omega / c`."""
        return self.angular_frequency / c

    @property
    def aperture(self) -> Float[Array, " "]:
        r"""The aperture :math:`A` of an isotropic antenna."""
        return self.wavelength**2 / (4 * jnp.pi)


class Antenna(BaseAntenna):
    """An antenna class, must be subclassed."""

    @property
    @abstractmethod
    def reference_power(self) -> Float[Array, " "]:
        r"""The reference power (W) radiated by this antenna.

        This is the maximal value of the pointing vector at a distance
        of one meter from this antenna, multiplied by the area of the sphere
        (:math:`4\phi`),
        to obtain a power.
        """

    @abstractmethod
    def fields(
        self,
        r: Float[ArrayLike, "*#batch 3"],
        t: Float[ArrayLike, "*#batch"] | None = None,
    ) -> tuple[Inexact[Array, "*batch 3"], Inexact[Array, "*batch 3"]]:
        r"""
        Compute electric and magnetic fields in vacuum at given position and (optional) time.

        Args:
            r: The array of positions.
            t: The array of time instants.

                If not provided, initial time instant
                is assumed.

        Returns:
            The electric :math:`\vec{E}` and magnetical :math:`\vec{B}` fields.

            Fields can be either real or complex-valued.
        """

    @eqx.filter_jit
    def pointing_vector(
        self,
        r: Float[ArrayLike, "*#batch 3"],
        t: Float[ArrayLike, "*#batch"] | None = None,
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
        return pointing_vector(e, b)

    def directivity(
        self,
        num_points: int = int(1e2),
    ) -> tuple[
        Float[Array, " 2*{num_points}"],
        Float[Array, " {num_points}"],
        Float[Array, "2*{num_points} {num_points}"],
    ]:
        """
        Compute an estimate of the antenna directivity for azimutal and elevation angles.

        .. note::

            Subclasses may provide a more accurate or exact
            implementation.

        Args:
            num_points: The number of points to sample along the elevation axis.

                Twice this number of points are sampled on the aximutal axis.

        Returns:
            Azimutal and elevation angles, as well as corresponding directivity values.

        .. seealso::

            :meth:`directive_gain`
        """
        u, du = jnp.linspace(0, 2 * jnp.pi, num_points * 2, retstep=True)
        v, dv = jnp.linspace(0, jnp.pi, num_points, retstep=True)
        x = jnp.outer(jnp.cos(u), jnp.sin(v))
        y = jnp.outer(jnp.sin(u), jnp.sin(v))
        z = jnp.outer(jnp.ones_like(u), jnp.cos(v))

        r = self.center + jnp.stack((x, y, z), axis=-1)

        s = self.pointing_vector(r)

        p = jnp.linalg.norm(s, axis=-1)

        ds = du * dv

        # Power per unit solid angle
        U = p / ds  # noqa: N806
        p_tot = jnp.sum(p * jnp.sin(v)) / (4 * jnp.pi)

        return u, v, U / p_tot

    def directive_gain(
        self,
        num_points: int = int(1e2),
    ) -> Float[Array, " "]:
        """
        Compute an estimate of the antenna directive gain.

        .. note::

            Subclasses may provide a more accurate or exact
            implementation.

        Args:
            num_points: The number of points used for the estimate.

        Returns:
            The antenna directive gain.

        .. seealso::

            :meth:`directivity`
        """
        return self.directivity(num_points=num_points)[-1].max()

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

        Returns:
            The resulting plot output.
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


class Dipole(Antenna):
    r"""
    A simple electrical (or Hertzian) dipole.

    Equations were obtained from :cite:`dipole,dipole-moment,dipole-antenna,directivity`, and assume
    a constant current across the dipole length.

    Args:
        frequency: The frequency at which the antenna is operating.
        num_wavelengths: The length of the dipole, relative to the wavelength.
        length: The absolute length of the dipole, supersedes ``num_wavelengths``.
        moment: The dipole moment.

            By default, the dipole is aligned with the z-axis.
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

            >>> from differt.em import Dipole
            >>>
            >>> ant = Dipole(frequency=1e9)
            >>> fig = ant.plot_radiation_pattern(backend="plotly")
            >>> fig  # doctest: +SKIP

        The second example shows how to plot the radiation
        pattern (antenna power) at 1 meter, but only
        in the x-z plane, for multiple dipole lengths.

        .. plot::

            >>> from differt.em import Dipole
            >>>
            >>> theta = jnp.linspace(0, 2 * jnp.pi, 200)
            >>> r = jnp.stack(
            ...     (jnp.cos(theta), jnp.zeros_like(theta), jnp.sin(theta)), axis=-1
            ... )
            >>> fig = plt.figure()
            >>> ax = fig.add_subplot(
            ...     projection="polar", facecolor="lightgoldenrodyellow"
            ... )
            >>> for ratio in [0.5, 1.0, 1.25, 1.5, 2.0]:
            ...     ant = Dipole(1e9, ratio)
            ...     power = jnp.linalg.norm(ant.pointing_vector(r), axis=-1)
            ...     _ = ax.plot(theta, power, label=rf"$\ell/\lambda = {ratio:1.2f}$")
            >>>
            >>> ax.tick_params(grid_color="palegoldenrod")
            >>> ax.set_rscale("log")
            >>> angle = jnp.deg2rad(-10)
            >>> ax.legend(  # doctest: +SKIP
            ...     loc="upper left",
            ...     bbox_to_anchor=(0.5 + jnp.cos(angle) / 2, 0.5 + jnp.sin(angle) / 2),
            ... )
            >>> plt.show()  # doctest: +SKIP
    """

    length: Float[Array, " "] = eqx.field(converter=jnp.asarray)
    """Dipole length (in meter)."""
    moment: Float[Array, "3"] = eqx.field(converter=jnp.asarray)
    """Dipole moment (in Coulomb-meter)."""

    def __init__(
        self,
        frequency: Float[ArrayLike, " "],
        num_wavelengths: Float[ArrayLike, " "] = 0.5,
        *,
        length: Float[ArrayLike, " "] | None = None,
        moment: Float[ArrayLike, "3"] | None = jnp.array([0.0, 0.0, 1.0]),
        current: Float[ArrayLike, " "] | None = 1.0,
        charge: Float[ArrayLike, " "] | None = None,
        center: Float[ArrayLike, "3"] = jnp.array([0.0, 0.0, 0.0]),
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

    @property
    def reference_power(self) -> Float[Array, " "]:
        p_0 = jnp.linalg.norm(self.moment)

        # Equivalent to
        # 4 * pi * (r=1) * mu_0 * self.angular_frequency**4 * p_0**2 / (16 * jnp.pi**2 * c)
        # but avoids overflow

        r = mu_0 * self.angular_frequency
        t = self.angular_frequency * p_0
        r *= t
        r *= t
        r *= self.angular_frequency / (4 * jnp.pi * c)

        return r

    @eqx.filter_jit
    def fields(
        self,
        r: Float[ArrayLike, "*#batch 3"],
        t: Float[ArrayLike, "*#batch"] | None = None,
    ) -> tuple[Inexact[Array, "*batch 3"], Inexact[Array, "*batch 3"]]:
        r = jnp.asarray(r)
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

        exp = (
            jnp.exp(j_k_r - 1j * w * jnp.asarray(t)[..., None])
            if t is not None
            else jnp.exp(j_k_r)
        )

        e *= exp
        b *= exp

        return e, b

    def directivity(
        self,
        num_points: int = int(1e2),
    ) -> tuple[
        Float[Array, " 2*{num_points}"],
        Float[Array, " {num_points}"],
        Float[Array, "2*{num_points} {num_points}"],
    ]:
        u = jnp.linspace(0, 2 * jnp.pi, num_points * 2)
        v = jnp.linspace(0, jnp.pi, num_points)
        x = jnp.outer(jnp.cos(u), jnp.sin(v))
        y = jnp.outer(jnp.sin(u), jnp.sin(v))
        z = jnp.outer(jnp.ones_like(u), jnp.cos(v))

        r = jnp.stack((x, y, z), axis=-1)

        p = self.moment / jnp.linalg.norm(self.moment)

        sin_theta = jnp.cross(r, p)

        return u, v, 1.5 * jax.lax.integer_pow(sin_theta, 2)

    def directive_gain(  # noqa: PLR6301
        self,
        num_points: int = int(1e2),  # noqa: ARG002
    ) -> Float[Array, " "]:
        return jnp.array(1.5)


class ShortDipole(Dipole):
    """Short dipole.

    Like :class:`Dipole`, but accounts for the fact that the current is not constant across the dipole length,
    which leads to more realistic results.

    However, fields are only derived for far field.

    Warning:
        Not implemented yed.
    """

    @eqx.filter_jit
    def fields(
        self,
        r: Float[ArrayLike, "*#batch 3"],
        t: Float[ArrayLike, "*#batch"] | None = None,
    ) -> tuple[Inexact[Array, "*batch 3"], Inexact[Array, "*batch 3"]]:
        raise NotImplementedError

    def directivity(
        self,
        num_points: int = int(1e2),
    ) -> tuple[
        Float[Array, " 2*{num_points}"],
        Float[Array, " {num_points}"],
        Float[Array, "2*{num_points} {num_points}"],
    ]:
        # Bypass Dipole's specialized implementation
        return Antenna.directivity(self, num_points=num_points)

    def directive_gain(
        self,
        num_points: int = int(1e2),
    ) -> Float[Array, " "]:
        # Bypass Dipole's specialized implementation
        return Antenna.directive_gain(self, num_points=num_points)


class RadiationPattern(BaseAntenna):
    """An antenna radiation pattern class, must be subclassed."""

    @abstractmethod
    def polarization_vectors(
        self,
        r: Float[ArrayLike, "*#batch 3"],
    ) -> tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]]:
        r"""
        Compute s and p polarization vectors.

        Args:
            r: The array of positions.

        Returns:
            The electric :math:`\vec{E}` and magnetical :math:`\vec{B}` fields.

            Fields can be either real or complex-valued.
        """

    def directivity(
        self,
        num_points: int = int(1e2),
    ) -> tuple[
        Float[Array, " 2*{num_points}"],
        Float[Array, " {num_points}"],
        Float[Array, "2*{num_points} {num_points}"],
    ]:
        """
        Compute an estimate of the antenna directivity for azimutal and elevation angles.

        .. note::

            Subclasses may provide a more accurate or exact
            implementation.

        Args:
            num_points: The number of points to sample along the elevation axis.

                Twice this number of points are sampled on the aximutal axis.

        Returns:
            Azimutal and elevation angles, as well as corresponding directivity values.

        .. seealso::

            :meth:`directive_gain`
        """
        u, _du = jnp.linspace(0, 2 * jnp.pi, num_points * 2, retstep=True)
        v, _dv = jnp.linspace(0, jnp.pi, num_points, retstep=True)
        x = jnp.outer(jnp.cos(u), jnp.sin(v))
        y = jnp.outer(jnp.sin(u), jnp.sin(v))
        z = jnp.outer(jnp.ones_like(u), jnp.cos(v))

        r = self.center + jnp.stack((x, y, z), axis=-1)

        s, p = self.polarization_vectors(r)

        g = dot(s) + dot(p)

        # TODO: check if this is correct

        return u, v, g

    def directive_gain(
        self,
        num_points: int = int(1e2),
    ) -> Float[Array, " "]:
        """
        Compute an estimate of the antenna directive gain.

        .. note::

            Subclasses may provide a more accurate or exact
            implementation.

        Args:
            num_points: The number of points used for the estimate.

        Returns:
            The antenna directive gain.

        .. seealso::

            :meth:`directivity`
        """
        return self.directivity(num_points=num_points)[-1].max()

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

        Returns:
            The resulting plot output.
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

        s = self.pointing_vector(r)  # type: ignore[reportAttributeAccessIssue]

        p = jnp.linalg.norm(s, axis=-1, keepdims=True)

        gain = p / p.max()

        r *= gain
        gain = jnp.squeeze(gain, axis=-1)

        return draw_surface(
            x=r[..., 0], y=r[..., 1], z=r[..., 2], colors=gain, **kwargs
        )


class HWDipolePattern(RadiationPattern):
    """An half-wave dipole radiation pattern."""

    direction: Float[Array, "3"] = eqx.field(converter=jnp.asarray)
    """The dipole direction."""

    def polarization_vectors(
        self,
        r: Float[ArrayLike, "*#batch 3"],
    ) -> tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]]:
        r = jnp.asarray(r)
        r_hat, r = normalize(r - self.center, keepdims=True)

        cos_theta = dot(r_hat, self.direction)
        sin_theta = jnp.sqrt(1 - cos_theta**2)

        d = 1.640922376984585  # Directive gain: 4 / Cin(2*pi)

        cos_theta = dot(d)
        sin_theta = jnp.sin(d)
        _d = safe_divide(jnp.cos(0.5 * jnp.pi * cos_theta), sin_theta)
        raise NotImplementedError


class ShortDipolePattern(RadiationPattern):
    """An short dipole radiation pattern."""

    direction: Float[Array, "3"] = eqx.field(converter=jnp.asarray)
    """The dipole direction."""
