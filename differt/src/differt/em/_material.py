# ruff:file-ignore[math-constant]

import dataclasses
import typing
from abc import abstractmethod
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from functools import partial
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

if TYPE_CHECKING:
    import differt_core.geometry

if TYPE_CHECKING or hasattr(typing, "GENERATING_DOCS"):
    from typing import Self
else:
    Self = Any  # Because runtime type checking from 'beartype' will fail when combined with 'jaxtyping'


class AbstractScatteringPattern(eqx.Module):
    """Abstract base class for diffuse scattering patterns."""

    @abstractmethod
    def __call__(
        self,
        k_i: Float[ArrayLike, "*#batch 3"],
        k_s: Float[ArrayLike, "*#batch 3"],
        n: Float[ArrayLike, "*#batch 3"],
    ) -> Float[Array, "*batch"]:
        r"""
        Compute the scattered power angular density.

        Args:
            k_i: The incident (unit) direction, from the previous path
                vertex to the interaction point.
            k_s: The scattered (unit) direction, from the interaction
                point to the next path vertex.
            n: The (unit) outward surface normal at the interaction point.

        Returns:
            The scattered power angular density, normalized so that its
            integral over the hemisphere around ``n`` is 1.
        """


class LambertianPattern(AbstractScatteringPattern):
    """The Lambertian (cosine) diffuse scattering pattern.

    This is the default :attr:`Material.scattering_pattern`.
    """

    def __call__(
        self,
        k_i: Float[ArrayLike, "*#batch 3"],
        k_s: Float[ArrayLike, "*#batch 3"],
        n: Float[ArrayLike, "*#batch 3"],
    ) -> Float[Array, "*batch"]:
        r"""
        Compute the Lambertian (cosine) diffuse scattering pattern.

        .. math::
            f_s(\hat{k}_s) = \frac{\max(\hat{n} \cdot \hat{k}_s, 0)}{\pi},

        normalized so that its integral over the hemisphere around
        :math:`\hat{n}` is 1. This pattern does not depend on the
        incident direction :math:`\hat{k}_i`.

        Args:
            k_i: The incident (unit) direction.
            k_s: The scattered (unit) direction.
            n: The (unit) surface normal.

        Returns:
            The scattered power angular density.
        """
        del k_i
        cos_theta_s = jnp.clip(
            jnp.sum(jnp.asarray(n) * jnp.asarray(k_s), axis=-1), 0.0, 1.0
        )
        return cos_theta_s / jnp.pi


def _specular_direction(
    k_i: Float[Array, "*batch 3"],
    n: Float[Array, "*batch 3"],
) -> Float[Array, "*batch 3"]:
    """Mirror-reflect the incident direction about the surface normal.

    Returns:
        The specular reflection direction.
    """
    return k_i - 2.0 * jnp.sum(k_i * n, axis=-1, keepdims=True) * n


def _normalized_lobe(
    k_s: Float[Array, "*batch 3"],
    axis: Float[Array, "*batch 3"],
    alpha: Float[ArrayLike, ""],
) -> Float[Array, "*batch"]:
    r"""
    Evaluate a directive power-cosine lobe centered on ``axis``.

    .. math::
        F_\alpha(\hat{k}_s, \hat{a}) = \frac{\alpha + 1}{4 \pi} \left(\frac{1 + \hat{a} \cdot \hat{k}_s}{2}\right)^\alpha,

    following Degli-Esposti's directive scattering model
    :cite:`rt-review`. Unlike
    :class:`LambertianPattern`, :math:`F_\alpha` has no closed-form
    normalization over just the hemisphere around an arbitrary surface
    normal (as opposed to around ``axis`` itself), so the constant
    :math:`(\alpha + 1) / (4\pi)` instead normalizes the integral of
    :math:`F_\alpha` over the *full* sphere centered on ``axis`` to 1.
    Provided ``axis`` lies well inside the physical hemisphere and
    :math:`\alpha` is not too small, only a small fraction of the lobe
    falls below the horizon, so the hemispherical integral remains close
    to (but, unlike the Lambertian pattern, not exactly) 1; expect
    visibly more energy than the Lambertian pattern for a small
    :math:`\alpha` or a near-grazing ``axis``.

    Returns:
        The lobe's angular density.
    """
    cos_psi = jnp.clip(jnp.sum(k_s * axis, axis=-1), -1.0, 1.0)
    alpha = jnp.asarray(alpha)
    c = (alpha + 1.0) / (4.0 * jnp.pi)
    return c * ((1.0 + cos_psi) / 2.0) ** alpha


class DirectivePattern(AbstractScatteringPattern):
    r"""
    A directive diffuse scattering pattern, with a lobe centered on the specular direction.

    Following Degli-Esposti's directive model :cite:`rt-review`
    (as also used by Sionna RT's ``DirectivePattern``), the scattered
    power is concentrated around the specular reflection direction
    :math:`\hat{k}_{sp}` (the mirror image of :math:`\hat{k}_i` about
    :math:`\hat{n}`), with :attr:`alpha_r` controlling how narrow the
    lobe is: :attr:`alpha_r` :math:`= 1` gives a broad lobe (similar in
    shape to, but distinct from, :class:`LambertianPattern`, since it is
    centered on :math:`\hat{k}_{sp}` rather than :math:`\hat{n}`), while
    a large :attr:`alpha_r` concentrates almost all power near the
    specular direction, approaching a purely specular reflection.

    See :math:`F_\alpha` below for the lobe formula and its
    normalization caveat.

    .. math::
        F_\alpha(\hat{k}_s, \hat{a}) = \frac{\alpha + 1}{4 \pi} \left(\frac{1 + \hat{a} \cdot \hat{k}_s}{2}\right)^\alpha,

    where :math:`\hat{a}` is the lobe's axis (here, the specular
    direction :math:`\hat{k}_{sp}`). Unlike :class:`LambertianPattern`,
    :math:`F_\alpha` has no closed-form normalization over just the
    hemisphere around an arbitrary surface normal (as opposed to around
    :math:`\hat{a}` itself), so the constant :math:`(\alpha + 1) / (4\pi)`
    instead normalizes the integral of :math:`F_\alpha` over the *full*
    sphere centered on :math:`\hat{a}` to 1. Provided :math:`\hat{a}` lies
    well inside the physical hemisphere and :math:`\alpha` is not too
    small, only a small fraction of the lobe falls below the horizon, so
    the hemispherical integral remains close to (but, unlike the
    Lambertian pattern, not exactly) 1; expect visibly more energy than
    the Lambertian pattern for a small :math:`\alpha` or a near-grazing
    :math:`\hat{a}`.
    """

    alpha_r: Float[ArrayLike, ""] = eqx.field(default=1.0)
    r"""The lobe width parameter :math:`\alpha_R \geq 1`; larger values give a narrower lobe."""

    def __call__(
        self,
        k_i: Float[ArrayLike, "*#batch 3"],
        k_s: Float[ArrayLike, "*#batch 3"],
        n: Float[ArrayLike, "*#batch 3"],
    ) -> Float[Array, "*batch"]:
        r"""
        Compute the directive diffuse scattering pattern.

        Args:
            k_i: The incident (unit) direction.
            k_s: The scattered (unit) direction.
            n: The (unit) surface normal.

        Returns:
            The scattered power angular density.
        """
        k_i, k_s, n = jnp.asarray(k_i), jnp.asarray(k_s), jnp.asarray(n)
        k_sp = _specular_direction(k_i, n)
        return _normalized_lobe(k_s, k_sp, self.alpha_r)


class BackscatteringPattern(AbstractScatteringPattern):
    r"""
    A diffuse scattering pattern combining a forward and a backscattering lobe.

    Following Degli-Esposti's combined scattering model
    :cite:`rt-review` (as also used by Sionna RT's
    ``BackscatteringPattern``), this mixes two directive lobes
    :math:`F_\alpha` (see :class:`DirectivePattern` for the lobe formula
    and its normalization caveat): one centered on the specular direction
    :math:`\hat{k}_{sp}` (like :class:`DirectivePattern`, with width
    :attr:`alpha_r`), and one centered on the retroreflection direction
    :math:`-\hat{k}_i` (back toward the transmitter, with width
    :attr:`alpha_i`), combined as

    .. math::
        f_s(\hat{k}_i, \hat{k}_s, \hat{n}) = \Lambda F_{\alpha_R}(\hat{k}_s, \hat{k}_{sp}) + (1 - \Lambda) F_{\alpha_I}(\hat{k}_s, -\hat{k}_i),

    where :math:`\Lambda \in [0, 1]` (:attr:`lambda_`) is the forward
    lobe's weight; :math:`\Lambda = 1` reduces to :class:`DirectivePattern`,
    while :math:`\Lambda = 0` gives a purely retroreflective (backscattering)
    lobe, e.g., as observed on rough surfaces exhibiting a strong return
    toward the transmitter (e.g., vegetation, some building facades).
    """

    alpha_r: Float[ArrayLike, ""] = eqx.field(default=1.0)
    r"""The forward-lobe width parameter :math:`\alpha_R \geq 1`."""
    alpha_i: Float[ArrayLike, ""] = eqx.field(default=1.0)
    r"""The backward-lobe width parameter :math:`\alpha_I \geq 1`."""
    lambda_: Float[ArrayLike, ""] = eqx.field(default=0.5)
    r"""The forward-lobe weight :math:`\Lambda \in [0, 1]`."""

    def __call__(
        self,
        k_i: Float[ArrayLike, "*#batch 3"],
        k_s: Float[ArrayLike, "*#batch 3"],
        n: Float[ArrayLike, "*#batch 3"],
    ) -> Float[Array, "*batch"]:
        r"""
        Compute the combined forward/backscattering diffuse scattering pattern.

        Args:
            k_i: The incident (unit) direction.
            k_s: The scattered (unit) direction.
            n: The (unit) surface normal.

        Returns:
            The scattered power angular density.
        """
        k_i, k_s, n = jnp.asarray(k_i), jnp.asarray(k_s), jnp.asarray(n)
        k_sp = _specular_direction(k_i, n)
        forward = _normalized_lobe(k_s, k_sp, self.alpha_r)
        backward = _normalized_lobe(k_s, -k_i, self.alpha_i)
        lambda_ = jnp.asarray(self.lambda_)
        return lambda_ * forward + (1.0 - lambda_) * backward


class Material(eqx.Module):
    """A class representing a material and its electrical properties.

    .. note::

        This class is also re-exported directly from the top-level :mod:`differt` package
        (e.g., ``from differt import Material``).
    """

    name: str = eqx.field(static=True)
    """
    The name of the material.
    """
    properties: Callable[
        [Float[ArrayLike, " *batch"]],
        tuple[Float[Array, " *batch"], Float[Array, " *batch"]],
    ] = eqx.field(static=True)
    """
    The callable that computes the electrical properties of the material at the given frequency.

    The signature of the callable must be as follows.

    Args:
        frequency: The frequency at which to compute the electrical properties.

    Returns:
        A tuple containing the relative permittivity and conductivity of the material.
    """
    thickness: Float[ArrayLike, ""] | None = eqx.field(default=None)
    """The thickness of the material."""
    scattering_coefficient: Float[ArrayLike, ""] = eqx.field(default=0.0)
    r"""
    The (rough-surface) scattering coefficient :math:`S \in [0, 1]`.

    The fraction :math:`S^2` of the reflected power is diverted to
    diffuse scattering (see
    :meth:`GeometricFieldSolver.scattering_matrix<differt.em.GeometricFieldSolver.scattering_matrix>`),
    with the remaining :math:`1 - S^2` staying specular. Defaults to
    ``0.0``, i.e., a perfectly smooth surface with no diffuse scattering,
    matching the behavior of materials that do not set this value
    explicitly.
    """
    xpd_coefficient: Float[ArrayLike, ""] = eqx.field(default=0.0)
    r"""
    The cross-polarization discrimination coefficient :math:`K_x \in [0, 1]` of the scattered field.

    The fraction :math:`K_x` of the diffusely-scattered energy is
    converted to the orthogonal polarization. Defaults to ``0.0``, i.e.,
    no cross-polarization.
    """
    scattering_pattern: AbstractScatteringPattern = eqx.field(
        default_factory=LambertianPattern
    )
    """
    The pattern that computes the angular density of the diffusely-scattered power.

    Used by
    :meth:`GeometricFieldSolver.scattering_matrix<differt.em.GeometricFieldSolver.scattering_matrix>`
    together with :attr:`scattering_coefficient`. Defaults to
    :class:`LambertianPattern`. A custom (e.g., directive) pattern can be
    provided instead, by subclassing :class:`AbstractScatteringPattern`.
    """
    aliases: tuple[str, ...] = eqx.field(default=(), static=True)
    """
    A tuple of name aliases for the material.
    """

    def __repr__(self) -> str:
        thickness_str = (
            f", thickness={self.thickness!r}" if self.thickness is not None else ""
        )
        aliases_str = f", aliases={self.aliases!r}" if self.aliases else ""
        return f"Material(name={self.name!r}{thickness_str}{aliases_str})"

    def relative_permittivity(
        self, frequency: Float[ArrayLike, " *batch"]
    ) -> Float[Array, " *batch"]:
        """
        Compute the relative permittivity of the material at the given frequency.

        Args:
            frequency: The frequency at which to compute the relative permittivity.

        Returns:
            The relative permittivity of the material.
        """
        return self.properties(frequency)[0]

    def conductivity(
        self, frequency: Float[ArrayLike, " *batch"]
    ) -> Float[Array, " *batch"]:
        """
        Compute the conductivity of the material at the given frequency.

        Args:
            frequency: The frequency at which to compute the conductivity.

        Returns:
            The conductivity of the material.
        """
        return self.properties(frequency)[1]

    @classmethod
    def from_itu_properties(
        cls,
        name: str,
        *itu_properties: tuple[
            Float[ArrayLike, ""],
            Float[ArrayLike, ""],
            Float[ArrayLike, ""],
            Float[ArrayLike, ""],
            tuple[Float[ArrayLike, ""], Float[ArrayLike, ""]] | None,
        ],
    ) -> Self:
        r"""
        Create a material from ITU properties.

        The ITU-R Recommendation P.2040-4 :cite:`itu-r-2040` defines the electrical properties of a material
        using 4 real-valued coefficients: **a**, **b**, **c**, and **c**. The :data:`materials` mapping
        is already populated with values from :cite:`itu-r-2040{Tab. 3}`.

        Args:
            name: The name of the material.
            itu_properties: Material properties and corresponding frequency range.

                Each tuple must contain:

                * **a** (:class:`Float[ArrayLike, '']<jaxtyping.Float>`):
                  The first coefficient for the real part of the relative permittivity.
                * **b** (:class:`Float[ArrayLike, '']<jaxtyping.Float>`):
                  The second coefficient for the real part of the relative permittivity.
                * **c** (:class:`Float[ArrayLike, '']<jaxtyping.Float>`):
                  The first coefficient for the conductivity.
                * **d** (:class:`Float[ArrayLike, '']<jaxtyping.Float>`):
                  The second coefficient for the conductivity.
                * **frequency_range**
                  (:class:`tuple`\[:class:`Float[ArrayLike, '']<jaxtyping.Float>`,
                  :class:`Float[ArrayLike, '']<jaxtyping.Float>`\]):
                  The frequency range (in GHz) for which the electrical
                  properties are assumed to be correct.

        Returns:
            The material with the given ITU properties.

        Raises:
            ValueError: If more than one frequency range is specified and one of them is ``None``.
        """
        if len(itu_properties) > 1 and any(prop[4] is None for prop in itu_properties):
            msg = "Only one frequency range can be used if 'None' is passed, as it will match any frequency"
            raise ValueError(msg)

        aliases = (f"itu_{name.lower().replace(' ', '_')}",)
        if len(itu_properties) == 1:
            a, b, c, d, f_range = itu_properties[0]

            def callback(
                freq: Float[ArrayLike, " *batch"],
                a: Float[ArrayLike, ""],
                b: Float[ArrayLike, ""],
                c: Float[ArrayLike, ""],
                d: Float[ArrayLike, ""],
                f_min_hz: float | None = f_range[0] * 1e9 if f_range else None,
                f_max_hz: float | None = f_range[1] * 1e9 if f_range else None,
            ) -> tuple[Float[Array, " *batch"], Float[Array, " *batch"]]:
                f_arr = jnp.asarray(freq)
                if f_min_hz is not None and f_max_hz is not None:
                    where = (f_min_hz <= f_arr) & (f_arr <= f_max_hz)
                else:
                    where = jnp.ones_like(f_arr, dtype=bool)

                freq_ghz = f_arr * 1e-9
                rel_perm = jnp.where(where, a * (freq_ghz**b), -1.0)
                cond = jnp.where(where, c * (freq_ghz**d), -1.0)
                return rel_perm, cond

            return cls(
                name=name,
                properties=partial(jax.jit(callback), a=a, b=b, c=c, d=d),
                aliases=aliases,
            )

        def properties(
            frequency: Float[ArrayLike, " *batch"],
        ) -> tuple[Float[Array, " *batch"], Float[Array, " *batch"]]:
            f_hz = jnp.asarray(frequency)
            batch = f_hz.shape
            f_hz_flat = f_hz.ravel()
            f_ghz_flat = f_hz_flat * 1e-9

            ranges_hz = [
                (prop[4][0] * 1e9, prop[4][1] * 1e9)
                if prop[4] is not None
                else (-jnp.inf, jnp.inf)
                for prop in itu_properties
            ]

            lower_bounds_hz = jnp.array([r[0] for r in ranges_hz])
            upper_bounds_hz = jnp.array([r[1] for r in ranges_hz])
            widths_hz = upper_bounds_hz - lower_bounds_hz

            # Generate masks for each frequency range (in Hz)
            masks = (f_hz_flat[:, None] >= lower_bounds_hz[None, :]) & (
                f_hz_flat[:, None] <= upper_bounds_hz[None, :]
            )

            # Some ranges overlap, e.g., a broad range with coarse coefficients
            # and a narrower range with more specific coefficients. When several
            # ranges match, prefer the narrowest (most specific) one.
            i_outside = len(itu_properties)  # Fallback index
            candidate_widths = jnp.where(masks, widths_hz[None, :], jnp.inf)
            indices = jnp.where(
                jnp.any(masks, axis=1),
                jnp.argmin(candidate_widths, axis=1),
                i_outside,
            )

            branches = [
                lambda f, a=prop[0], b=prop[1], c=prop[2], d=prop[3]: (
                    a * (f**b),
                    c * (f**d),
                )
                for prop in itu_properties
            ]
            branches.append(
                lambda f: (
                    -jnp.ones_like(f),
                    -jnp.ones_like(f),
                )
            )

            rel_perm, cond = jax.vmap(
                lambda freq, idx: jax.lax.switch(
                    idx,
                    branches,
                    freq,
                ),
            )(f_ghz_flat, indices)

            return rel_perm.reshape(batch), cond.reshape(batch)

        return cls(
            name=name,
            properties=properties,
            aliases=aliases,
        )


class MaterialsDict(dict[str, Material]):  # ruff: ignore[subclass-builtin]
    """A dictionary subclass mapping material names to material instances with automatic alias support.

    This dictionary stores materials keyed by their primary name (:attr:`Material.name`).
    Indexing, membership checks (``in``), getting, and deletion using any material
    alias (defined in :attr:`Material.aliases`) automatically resolve to the primary material.
    """

    def __init__(
        self,
        other: Mapping[str, Material] | Iterable[Material | tuple[str, Material]] = (),
        /,
        **kwargs: Material,
    ) -> None:
        super().__init__()
        self.update(other, **kwargs)

    def __hash__(self) -> int:  # type: ignore[override]
        # 'dict' is unhashable, but this mapping is used as (conceptually
        # immutable) static data on 'GeometricFieldSolver.radio_materials',
        # which must be hashable to be usable as a JIT static argument/aux-data.
        return hash(tuple(sorted(self.items())))

    def _resolve(self, key: Any) -> Any:
        """Return the primary key that ``key`` (a name or alias) maps to, or ``key`` unchanged if unknown."""
        if not isinstance(key, str) or super().__contains__(key):
            return key
        return next((name for name, mat in self.items() if key in mat.aliases), key)

    def __missing__(self, key: str) -> Material:
        real_key = self._resolve(key)
        if real_key == key:
            raise KeyError(key)
        return self[real_key]

    def __contains__(self, key: object) -> bool:
        return super().__contains__(self._resolve(key))

    def __delitem__(self, key: str) -> None:
        super().__delitem__(self._resolve(key))

    def __setitem__(self, key: str, value: Material) -> None:
        real_key = self._resolve(key)
        if super().__contains__(real_key):
            super().__setitem__(real_key, value)
        elif isinstance(value, Material):
            super().__setitem__(value.name, value)
        else:
            super().__setitem__(key, value)

    def get(self, key: object, default: Any = None) -> Any:
        return super().get(self._resolve(key), default)

    def pop(self, key: object, *default: Any) -> Any:
        real_key = self._resolve(key)
        if super().__contains__(real_key):
            return super().pop(real_key)
        if default:
            return default[0]
        raise KeyError(key)

    def setdefault(self, key: str, default: Any = None) -> Any:
        real_key = self._resolve(key)
        if super().__contains__(real_key):
            return self[real_key]
        self[key] = default
        return default

    def update(self, other: Any = (), /, **kwargs: Material) -> None:
        items: Iterable[Any] = other.items() if isinstance(other, Mapping) else other
        for item in items:
            if isinstance(item, Material):
                self[item.name] = item
            else:
                key, value = item
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value


# ITU-R P.2040-4 materials from Table 3.
#
# This table is kept separately from `Material`, which is ITU-agnostic, so that
# the raw per-frequency-range coefficients remain available (e.g., to generate the
# documentation table) without `Material` having to store them.
_ITU_MATERIALS_TABLE: dict[
    str,
    tuple[
        tuple[
            Any,
            Any,
            Any,
            Any,
            tuple[Any, Any] | None,
        ],
        ...,
    ],
] = {}


def _add_material(
    name: str,
    *itu_properties: tuple[
        Any,
        Any,
        Any,
        Any,
        tuple[Any, Any] | None,
    ],
) -> Material:
    _ITU_MATERIALS_TABLE[name] = itu_properties
    return Material.from_itu_properties(name, *itu_properties)


_materials = [
    _add_material("Vacuum", (1.0, 0.0, 0.0, 0.0, None)),
    _add_material(
        "Concrete",
        (5.24, 0.0, 0.0462, 0.7822, (1.0, 100.0)),
        (5.17, 0.0, 0.0145, 1.09, (110.0, 330.0)),
    ),
    _add_material(
        "Brick",
        (3.91, 0.0, 0.0238, 0.16, (1.0, 40.0)),
        (4.15, 0.0, 0.0006, 1.5712, (110.0, 330.0)),
    ),
    _add_material(
        "Plasterboard",
        (2.73, 0.0, 0.0085, 0.9395, (1.0, 100.0)),
        (2.56, 0.0, 0.0001, 1.7799, (110.0, 330.0)),
        (2.65, 0.0, 0.0002, 1.598, (100.0, 400.0)),
    ),
    _add_material(
        "Wood",
        (1.99, 0.0, 0.0047, 1.0718, (0.001, 100.0)),
        (1.82, 0.0, 0.0040, 1.0761, (110.0, 330.0)),
        (2.1183, 0.0, 0.0055, 1.1113, (100.0, 400.0)),
    ),
    _add_material(
        "Glass",
        (6.31, 0.0, 0.0036, 1.3394, (0.1, 100.0)),
        (6.5767, 0.0, 0.0012, 1.4697, (100.0, 400.0)),
        (5.79, 0.0, 0.0004, 1.658, (220.0, 450.0)),
    ),
    _add_material(
        "Clear Acrylic",
        (2.58, 0.0, 0.0001, 1.6524, (110.0, 330.0)),
    ),
    _add_material(
        "Ceiling board",
        (1.48, 0.0, 0.0011, 1.0750, (1.0, 100.0)),
        (1.52, 0.0, 0.0029, 1.029, (220.0, 450.0)),
        (1.2567, 0.0, 0.00013, 1.454, (100.0, 400.0)),
    ),
    _add_material(
        "Chipboard",
        (2.58, 0.0, 0.0217, 0.7800, (1.0, 100.0)),
        (2.16, 0.0, 0.0023, 1.359, (100.0, 200.0)),
    ),
    _add_material(
        "Plywood",
        (2.71, 0.0, 0.33, 0.0, (1.0, 40.0)),
        (1.94, 0.0, 0.0067, 0.9982, (110.0, 330.0)),
        (2.17, 0.0, 0.0063, 1.045, (100.0, 400.0)),
    ),
    _add_material(
        "Marble",
        (7.074, 0.0, 0.0055, 0.9262, (1.0, 60.0)),
        (7.94, 0.0, 0.0001, 1.7330, (110.0, 330.0)),
        (8.62, 0.0, 0.0027, 1.15, (100.0, 400.0)),
    ),
    _add_material(
        "Floorboard",
        (3.66, 0.0, 0.0044, 1.3515, (50.0, 100.0)),
        (5.27, 0.0, 2.22e-17, 7.3413, (220.0, 300.0)),
        (5.27, 0.0, 0.0003, 2.0298, (300.0, 400.0)),
        (5.27, 0.0, 49.8726, 0.0, (400.0, 450.0)),
        (3.1575, 0.0, 0.001675, 1.32775, (100.0, 400.0)),
    ),
    _add_material(
        "Vinyl tile",
        (3.62, 0.0, 0.0051, 0.8422, (1.0, 40.0)),
    ),
    _add_material(
        "Carpet tile",
        (2.08, 0.0, 0.0009, 0.8200, (1.0, 40.0)),
    ),
    _add_material(
        "Asphalt concrete",
        (4.83, 0.0, 0.0108, 1.3969, (1.0, 40.0)),
    ),
    _add_material("Metal", (1.0, 0.0, 1e7, 0.0, (1.0, 100.0))),
    _add_material("Very dry ground", (3.0, 0.0, 0.00015, 2.52, (1.0, 10.0))),
    _add_material("Medium dry ground", (15.0, -0.1, 0.035, 1.63, (1.0, 10.0))),
    _add_material("Wet ground", (30.0, -0.4, 0.15, 1.30, (1.0, 10.0))),
]

materials: MaterialsDict = MaterialsDict(_materials)
"""A dictionary mapping material names and their aliases to corresponding :class:`Material` instances.

For convenience, each material can be accessed either by its official ITU name (e.g., ``'Concrete'``)
or by its Sionna-compatible alias (e.g., ``'itu_concrete'``).

See :ref:`itu-materials-table` for a table of all available ITU radio materials, their aliases, and electrical properties."""

del _materials


def _populate_materials(
    scene_materials: Iterable["differt_core.geometry.Material"],
    materials: MutableMapping[str, Material],
) -> None:
    """
    Populate ``materials`` in place with the per-shape radio material overrides of a loaded XML scene.

    Used by :meth:`Scene.load_xml<differt.geometry.Scene.load_xml>` so that a scene's ITU
    radio materials (currently, only ``thickness``, read from a
    ``<float name="thickness" value="..."/>`` child element) are ready to use, without the
    caller having to build a materials mapping by hand.

    Each overridden material is looked up in ``materials`` by its generic,
    ITU-type-derived name (e.g., ``'itu_glass'``) to find its base electrical properties,
    then stored back keyed the same way -- *unless* another material of the same type in
    ``scene_materials`` disagrees on the override (e.g., two ``type="glass"`` shapes with
    different ``thickness`` values), in which case each is kept under its own, unique XML id
    instead, so they remain distinguishable. This matches how
    :meth:`Scene.load_xml<differt.geometry.Scene.load_xml>` keys
    :attr:`Mesh.material_names<differt.geometry.Mesh.material_names>`.

    Args:
        scene_materials: The materials of a loaded scene, e.g., the values of
            ``differt_core.geometry.SionnaScene.load_xml(file).materials``.
        materials: The mapping to populate, both as the source of each material's base
            electrical properties (matched by :attr:`Material.name<differt.em._material.Material.name>`,
            or one of its :attr:`Material.aliases<differt.em._material.Material.aliases>`) and as the
            target it is updated in. A material whose name does not match any entry (e.g.,
            a non-ITU, purely visual material) is skipped.

    Raises:
        ValueError: If a material is already present in ``materials`` with a different,
            already-set ``thickness`` (e.g., from an earlier call with a conflicting scene).
    """
    scene_materials = list(scene_materials)

    seen: dict[str, float | None] = {}
    non_uniform_names: set[str] = set()
    for mat in scene_materials:
        if mat.name in seen:
            if seen[mat.name] != mat.thickness:
                non_uniform_names.add(mat.name)
        else:
            seen[mat.name] = mat.thickness

    for mat in scene_materials:
        if mat.thickness is None:
            continue

        base = materials.get(mat.name)
        if base is None:
            continue

        is_non_uniform = mat.name in non_uniform_names
        key = mat.id if is_non_uniform else mat.name
        existing = materials.get(key)

        if (
            existing is not None
            and existing.thickness is not None
            and existing.thickness != mat.thickness
        ):
            msg = (
                f"Material {key!r} is already present with a different 'thickness' "
                f"({existing.thickness!r} != {mat.thickness!r})."
            )
            raise ValueError(msg)

        value = existing if existing is not None else base
        if is_non_uniform:
            # A brand new key that does not match 'value.name' (nor any of its
            # aliases) would otherwise be silently stored under 'value.name' by
            # 'MaterialsDict.__setitem__', so the material must be renamed to
            # its unique id to be kept distinguishable. 'name' is a static
            # field, so it must be replaced via 'dataclasses.replace' rather
            # than 'eqx.tree_at', which only rewrites pytree leaves.
            value = dataclasses.replace(value, name=key)
        materials[key] = eqx.tree_at(
            lambda m: m.thickness,
            value,
            replace=mat.thickness,
            is_leaf=lambda x: x is None,
        )
