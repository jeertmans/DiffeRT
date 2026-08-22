from abc import abstractmethod
from dataclasses import KW_ONLY
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Inexact

from differt.geometry._utils import (
    cartesian_to_spherical,
    normalize,
    spherical_to_cartesian,
)
from differt.plotting import PlotOutput, draw_surface
from differt.utils import safe_divide

from ._constants import c, epsilon_0, mu_0


@jax.jit
def poynting_vector(
    e: Inexact[ArrayLike, "*#batch 3"],
    b: Inexact[ArrayLike, "*#batch 3"],
) -> Inexact[Array, "*batch 3"]:
    r"""
    Compute the Poynting vector in vacuum from electric :math:`\vec{E}` and magnetic :math:`\vec{B}` fields.

    Args:
        e: The electrical field.
        b: The magnetic field.

    Returns:
        The Poynting vector :math:`\vec{S}`.

        It can be either real or complex-valued.
    """
    return jnp.cross(jnp.asarray(e), jnp.asarray(b)) / mu_0


class BaseAntenna(eqx.Module):
    """An antenna class, base class for :class:`AbstractAntenna` and :class:`AbstractRadiationPattern`."""

    frequency: Float[Array, ""]
    """The frequency :math:`f` at which the antenna is operating."""
    _: KW_ONLY
    center: Float[Array, "3"] = eqx.field(
        default_factory=lambda: jnp.array([0.0, 0.0, 0.0])
    )
    """The center position of the antenna, from which the fields are radiated.

    Default value is the origin.
    """

    @property
    def period(self) -> Float[Array, ""]:
        """The period :math:`T = 1/f`."""
        return 1 / self.frequency

    @property
    def angular_frequency(self) -> Float[Array, ""]:
        r"""The angular frequency :math:`\omega = 2 \pi f`."""
        return 2 * jnp.pi * self.frequency

    @property
    def wavelength(self) -> Float[Array, ""]:
        r"""The wavelength :math:`\lambda = c / f`."""
        return c * self.period

    @property
    def wavenumber(self) -> Float[Array, ""]:
        r"""The wavenumber :math:`k = \omega / c`."""
        return self.angular_frequency / c

    @property
    def aperture(self) -> Float[Array, ""]:
        r"""The aperture :math:`A` of an isotropic antenna."""
        return self.wavelength**2 / (4 * jnp.pi)


class AbstractAntenna(BaseAntenna):
    """Abstract base class for antennas."""

    @property
    @abstractmethod
    def reference_power(self) -> Float[Array, ""]:
        r"""The reference power (W) radiated by this antenna.

        This is the maximal value of the Poynting vector at a distance
        of one meter from this antenna, multiplied by the area of the sphere
        (:math:`4\pi`),
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
            r: Position vector relative to the antenna center.
            t: Time instant.

                If not provided, initial time instant
                is assumed.

        Returns:
            The electric :math:`\vec{E}` and magnetic :math:`\vec{B}` fields.

            Fields can be either real or complex-valued.
        """

    def wavefront_radii(  # ruff:ignore[no-self-use]
        self,
        k_hat: Float[ArrayLike, "*#batch 3"],
    ) -> (
        Float[Array, "*batch"]
        | tuple[
            Float[Array, "*batch"],
            Float[Array, "*batch 3"],
            Float[Array, "*batch"],
            Float[Array, "*batch 3"],
        ]
        | None
    ):
        r"""
        Return the radii of curvature of the wavefront this antenna emits, in the given direction(s).

        This is an *offset* distance (or pair of distances) -- the same
        quantity as, and with the same sign convention as,
        ``tx_wavefront_radii`` -- added to the geometric path length by
        :meth:`GeometricFieldSolver.spreading_factor
        <differt.em.GeometricFieldSolver.spreading_factor>`, rather than
        the (generally much larger, and direction-*independent* only for
        an isotropic source) total radius of curvature at some distant
        observation point. :meth:`GeometricFieldSolver.compute_fields
        <differt.em.GeometricFieldSolver.compute_fields>` calls this with
        the direction of the first path segment (leaving the
        transmitter) whenever ``tx_polarization`` is set to an
        :class:`AbstractAntenna` instance, using the result *instead of* the
        solver's own ``tx_wavefront_radii`` attribute.

        A single position-like argument cannot, by itself, encode an
        astigmatic wavefront's orientation (its two principal radii apply
        along two specific, generally direction-dependent axes, not just
        "the wavefront's curvature at that point"); this is why the
        argument here is the propagation *direction* rather than a
        position -- see the ``(rho_s, s_hat, rho_p, p_hat)`` return case
        below.

        Args:
            k_hat: The (unit-length) direction of propagation away from
                this antenna, towards the observation point.

        Returns:
            :data:`None` for a planar wavefront (the far-field, plane-wave
            approximation -- see :class:`AbstractFarFieldAntenna`); a single value
            for an isotropic (spherical) wavefront, the same in every
            direction; or a ``(rho_s, s_hat, rho_p, p_hat)`` 4-tuple for a
            general astigmatic wavefront, where ``rho_s`` and ``rho_p``
            are the two principal radii and ``s_hat``/``p_hat`` are unit
            vectors (orthogonal to ``k_hat`` and to each other) giving
            the direction each one applies along -- the same (explicit
            vector pair, rather than a bare angle) convention used
            elsewhere in this module for a local s-/p-plane basis, see
            :func:`sp_directions<differt.em.sp_directions>` and
            :func:`sp_rotation_matrix<differt.em.sp_rotation_matrix>`
            (which a subclass reporting an astigmatic wavefront can reuse
            to derive ``s_hat``/``p_hat`` from ``k_hat`` and its own
            fixed reference axis, e.g. :attr:`Dipole.moment`, exactly as
            :func:`sp_directions<differt.em.sp_directions>` derives a
            local basis from a ray direction and a surface normal).

            The default implementation returns ``0`` for every direction,
            i.e., an ideal point source located exactly at this
            antenna's :attr:`~BaseAntenna.center` (no offset at all).
        """
        return jnp.zeros(jnp.shape(jnp.asarray(k_hat))[:-1])

    @eqx.filter_jit
    def poynting_vector(
        self,
        r: Float[ArrayLike, "*#batch 3"],
        t: Float[ArrayLike, "*#batch"] | None = None,
    ) -> Inexact[Array, "*batch 3"]:
        r"""
        Compute the Poynting vector in vacuum at given position and (optional) time.

        Args:
            r: Position vector relative to the antenna center.
            t: Time instant.

                If not provided, initial time instant
                is assumed.

        Returns:
            The Poynting vector :math:`\vec{S}`.

            It can be either real or complex-valued.
        """
        e, b = self.fields(r, t)
        return poynting_vector(e, b)

    def directivity(
        self,
        num_points: int = int(1e2),
    ) -> tuple[
        Float[Array, " 2*{num_points}"],
        Float[Array, " {num_points}"],
        Float[Array, "2*{num_points} {num_points}"],
    ]:
        """
        Compute an estimate of the antenna directivity for azimuthal and elevation angles.

        .. note::

            Subclasses may provide a more accurate or exact
            implementation.

        Args:
            num_points: The number of points to sample along the elevation axis.

                Twice this number of points are sampled on the azimuthal axis.

        Returns:
            Azimuthal and elevation angles, as well as corresponding directivity values.

        .. seealso::

            :meth:`directive_gain`
        """
        u, du = jnp.linspace(0, 2 * jnp.pi, num_points * 2, retstep=True)
        v, dv = jnp.linspace(0, jnp.pi, num_points, retstep=True)
        x = jnp.outer(jnp.cos(u), jnp.sin(v))
        y = jnp.outer(jnp.sin(u), jnp.sin(v))
        z = jnp.outer(jnp.ones_like(u), jnp.cos(v))

        r = self.center + jnp.stack((x, y, z), axis=-1)

        s = self.poynting_vector(r)

        p = jnp.linalg.norm(s, axis=-1)

        ds = du * dv

        # Power per unit solid angle
        U = p / ds  # ruff:ignore[non-lowercase-variable-in-function]
        p_tot = jnp.sum(p * jnp.sin(v)) / (4 * jnp.pi)

        return u, v, U / p_tot

    def directive_gain(
        self,
        num_points: int = int(1e2),
    ) -> Float[Array, ""]:
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
        distance: Float[ArrayLike, ""] = 1.0,
        num_wavelengths: Float[ArrayLike, ""] | None = None,
        **kwargs: Any,
    ) -> PlotOutput:
        """
        Plot the radiation pattern (normalized power) of this antenna.

        The power is computed on points on a sphere around the antenna.

        Args:
            num_points: The number of points to sample along the elevation axis.

                Twice this number of points are sampled on the azimuthal axis.
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

        s = self.poynting_vector(r)

        p = jnp.linalg.norm(s, axis=-1, keepdims=True)

        gain = p / p.max()

        r = self.center + (r - self.center) * gain
        gain = jnp.squeeze(gain, axis=-1)

        return draw_surface(
            x=r[..., 0], y=r[..., 1], z=r[..., 2], colors=gain, **kwargs
        )


class Dipole(AbstractAntenna):
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

            If this is provided, which is the default, only the direction of the moment
            vector is used, and its intensity is set to match the dipole moment with
            specified current.
        charge: The dipole charge (in Coulomb), assuming opposite charges on either ends of the dipole.

            If this is provided, this takes precedence over ``current``.
        center: The center position of the antenna, from which the fields are radiated.
        look_at: When provided, re-orient the antenna to look at the given point.

            This overrides the direction of the dipole moment.

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
            >>> for ratio in [0.5, 1.0, 1.25, 1.5, 2.0]:  # doctest: +SKIP
            ...     ant = Dipole(1e9, ratio)
            ...     power = jnp.linalg.norm(ant.poynting_vector(r), axis=-1)
            ...     ax.plot(theta, power, label=rf"$\ell/\lambda = {ratio:1.2f}$")
            >>>
            >>> ax.tick_params(grid_color="palegoldenrod")
            >>> ax.set_rscale("log")
            >>> angle = jnp.deg2rad(-10)
            >>> ax.legend(  # doctest: +SKIP
            ...     loc="upper left",
            ...     bbox_to_anchor=(0.5 + jnp.cos(angle) / 2, 0.5 + jnp.sin(angle) / 2),
            ... )
            >>> plt.show()  # doctest: +SKIP

        The third example shows how to orient the antenna to look at a given point.

        .. plotly::
            :fig-vars: fig

            >>> from differt.em import Dipole
            >>>
            >>> ant = Dipole(frequency=1e9, look_at=jnp.array([0.0, -1.0, -1.0]))
            >>> fig = ant.plot_radiation_pattern(backend="plotly")
            >>> fig  # doctest: +SKIP
    """

    length: Float[Array, ""]
    """Dipole length (in meter)."""
    moment: Float[Array, "3"]
    """Dipole moment (in Coulomb-meter)."""

    def __init__(
        self,
        frequency: Float[ArrayLike, ""],
        num_wavelengths: Float[ArrayLike, ""] = 0.5,
        *,
        length: Float[ArrayLike, ""] | None = None,
        moment: Float[ArrayLike, "3"] | None = jnp.array([0.0, 0.0, 1.0]),
        current: Float[ArrayLike, ""] | None = 1.0,
        charge: Float[ArrayLike, ""] | None = None,
        center: Float[ArrayLike, "3"] = jnp.array([0.0, 0.0, 0.0]),
        look_at: Float[ArrayLike, "3"] | None = None,
    ) -> None:
        super().__init__(jnp.asarray(frequency), center=jnp.asarray(center))

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

        if look_at is not None:
            moment = spherical_to_cartesian(
                cartesian_to_spherical(moment)
                + (
                    cartesian_to_spherical(
                        normalize(jnp.asarray(look_at) - self.center)[0]
                    )
                    - cartesian_to_spherical(jnp.array([1.0, 0.0, 0.0]))
                )
            )

        self.moment = moment

    @property
    def reference_power(self) -> Float[Array, ""]:
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

    def directive_gain(  # ruff:ignore[no-self-use]
        self,
        num_points: int = int(1e2),  # ruff:ignore[unused-method-argument]
    ) -> Float[Array, ""]:
        return jnp.array(1.5)


class ShortDipole(Dipole):
    """Short dipole.

    Like :class:`Dipole`, but accounts for the fact that the current is not constant across the dipole length,
    which leads to more realistic results.

    However, fields are only derived for far field.

    Warning:
        Not implemented yet.
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
        return AbstractAntenna.directivity(self, num_points=num_points)

    def directive_gain(
        self,
        num_points: int = int(1e2),
    ) -> Float[Array, ""]:
        # Bypass Dipole's specialized implementation
        return AbstractAntenna.directive_gain(self, num_points=num_points)


class AbstractFarFieldAntenna(AbstractAntenna):
    """
    Abstract base class for antennas that are only ever used in the far field.

    Subclass this (instead of :class:`AbstractAntenna`) when your antenna's
    intended use is the far field, i.e., a locally planar wavefront:
    :meth:`wavefront_radii` is implemented once and for all here, always
    returning :data:`None`, so subclasses only need to implement
    :meth:`~AbstractAntenna.fields` and :attr:`~AbstractAntenna.reference_power` (exactly
    like a plain :class:`AbstractAntenna`).

    When used as ``tx_polarization`` in
    :meth:`GeometricFieldSolver.compute_fields
    <differt.em.GeometricFieldSolver.compute_fields>`, this makes the
    solver treat the source as an ideal point source infinitely far away
    (plane-wave incidence), regardless of whatever ``tx_wavefront_radii``
    attribute the solver was configured with -- see
    :meth:`AbstractAntenna.wavefront_radii` and
    :meth:`GeometricFieldSolver.spreading_factor
    <differt.em.GeometricFieldSolver.spreading_factor>`.
    """

    def wavefront_radii(  # ruff:ignore[no-self-use]
        self,
        k_hat: Float[  # ruff:ignore[unused-method-argument]
            ArrayLike, "*#batch 3"
        ],
    ) -> None:
        """Return :data:`None` (a planar wavefront), unconditionally."""
        return


class FarFieldDipoleAntenna(AbstractFarFieldAntenna, Dipole):
    """
    A :class:`Dipole` restricted to the far field (a planar, rather than spherical, wavefront).

    Otherwise identical to :class:`Dipole` (same constructor, same
    :meth:`~AbstractAntenna.fields`); only :meth:`~AbstractFarFieldAntenna.wavefront_radii`
    differs, see :class:`AbstractFarFieldAntenna`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        Dipole.__init__(self, *args, **kwargs)


class AbstractRadiationPattern(BaseAntenna):
    """Abstract base class for antenna radiation patterns."""

    @abstractmethod
    def polarization_vectors(
        self,
        r: Float[ArrayLike, "*#batch 3"],
    ) -> tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]]:
        r"""
        Compute s and p polarization vectors.

        Args:
            r: Position vector relative to the antenna center.

        Returns:
            The s and p polarization unit vectors.
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
        Compute an estimate of the antenna directivity for azimuthal and elevation angles.

        .. note::

            Subclasses may provide a more accurate or exact
            implementation.

        Args:
            num_points: The number of points to sample along the elevation axis.

                Twice this number of points are sampled on the azimuthal axis.

        Returns:
            Azimuthal and elevation angles, as well as corresponding directivity values.

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

        g = jnp.sum(s * s, axis=-1) + jnp.sum(p * p, axis=-1)

        # TODO: check if this is correct

        return u, v, g

    def directive_gain(
        self,
        num_points: int = int(1e2),
    ) -> Float[Array, ""]:
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
        distance: Float[ArrayLike, ""] = 1.0,
        num_wavelengths: Float[ArrayLike, ""] | None = None,
        **kwargs: Any,
    ) -> PlotOutput:
        """
        Plot the radiation pattern (normalized power) of this antenna.

        The power is computed on points on a sphere around the antenna.

        Args:
            num_points: The number of points to sample along the elevation axis.

                Twice this number of points are sampled on the azimuthal axis.
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

        s, p = self.polarization_vectors(r)

        power = jnp.sum(s * s, axis=-1, keepdims=True) + jnp.sum(
            p * p, axis=-1, keepdims=True
        )

        gain = power / power.max()

        r *= gain
        gain = jnp.squeeze(gain, axis=-1)

        return draw_surface(
            x=r[..., 0], y=r[..., 1], z=r[..., 2], colors=gain, **kwargs
        )


class HWDipolePattern(AbstractRadiationPattern):
    """A half-wave dipole radiation pattern."""

    direction: Float[Array, "3"]
    """The dipole direction."""

    def polarization_vectors(
        self,
        r: Float[ArrayLike, "*#batch 3"],
    ) -> tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]]:
        r = jnp.asarray(r)
        r_hat, r = normalize(r - self.center, keepdims=True)

        cos_theta = jnp.sum(r_hat * self.direction, axis=-1)
        sin_theta = jnp.sqrt(1 - cos_theta**2)

        d = 1.640922376984585  # Directive gain: 4 / Cin(2*pi)

        cos_theta = jnp.sum(d * d, axis=-1)
        sin_theta = jnp.sin(d)
        _d = safe_divide(jnp.cos(0.5 * jnp.pi * cos_theta), sin_theta)
        raise NotImplementedError


class ShortDipolePattern(AbstractRadiationPattern):
    """A short dipole radiation pattern."""

    direction: Float[Array, "3"]
    """The dipole direction."""
