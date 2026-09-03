from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, ArrayLike, Float, Inexact, PRNGKeyArray

from differt.em import c
from differt.em._antenna import (
    AbstractAntenna,
    AbstractFarFieldAntenna,
    AbstractRadiationPattern,
    Dipole,
    FarFieldDipoleAntenna,
    HWDipolePattern,
    ShortDipole,
)
from differt.geometry import normalize, spherical_to_cartesian

from ..plotting.params import (
    skip_if_matplotlib_not_installed,
    skip_if_plotly_not_installed,
    skip_if_vispy_not_installed,
)


@pytest.fixture
def antenna() -> Dipole:
    return Dipole(
        frequency=1e9,
    )


class _IsotropicAntenna(AbstractAntenna):
    """A minimal concrete antenna with direction-independent fields.

    Used to exercise :class:`AbstractAntenna`'s generic (non-overridden)
    :meth:`~AbstractAntenna.directivity` and
    :meth:`~AbstractAntenna.directive_gain` estimators, which
    :class:`Dipole` (and its subclasses) override with a closed-form
    expression.
    """

    @property
    def reference_power(self) -> Float[Array, ""]:
        return jnp.array(1.0)

    def fields(
        self,
        r: Float[ArrayLike, "*#batch 3"],
        t: Float[ArrayLike, "*#batch"]  # ruff:ignore[unused-method-argument]
        | None = None,
    ) -> tuple[Inexact[Array, "*batch 3"], Inexact[Array, "*batch 3"]]:
        r = jnp.asarray(r)
        e = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), r.shape)
        b = jnp.broadcast_to(jnp.array([0.0, 1.0, 0.0]), r.shape)
        return e, b


class _ConstantPolarizationPattern(AbstractRadiationPattern):
    """A minimal concrete radiation pattern with a simple, direction-dependent
    (but not physically meaningful) pair of polarization vectors.

    Used to exercise :class:`AbstractRadiationPattern`'s generic
    :meth:`~AbstractRadiationPattern.directivity`,
    :meth:`~AbstractRadiationPattern.directive_gain`, and
    :meth:`~AbstractRadiationPattern.plot_radiation_pattern`
    implementations.

    The vectors vary with direction (rather than being constant) so that
    the resulting normalized colors in :meth:`~AbstractRadiationPattern.plot_radiation_pattern`
    are not degenerate (a constant color triggers a spurious
    divide-by-zero when Matplotlib normalizes it).
    """

    def polarization_vectors(
        self,
        r: Float[ArrayLike, "*#batch 3"],
    ) -> tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]]:
        r_hat, _ = normalize(jnp.asarray(r) - self.center, keepdims=True)
        s = r_hat * jnp.array([1.0, 0.0, 0.0])
        p = r_hat * jnp.array([0.0, 1.0, 0.0])
        return s, p


class TestAntenna:
    def test_frequency(self, antenna: AbstractAntenna) -> None:
        chex.assert_trees_all_equal(antenna.frequency, 1e9)

    def test_center(self, antenna: AbstractAntenna) -> None:
        chex.assert_trees_all_equal(antenna.center, jnp.zeros(3))

    def test_period(self, antenna: AbstractAntenna) -> None:
        chex.assert_trees_all_close(antenna.period, 1 / 1e9)

    def test_angular_frequency(self, antenna: AbstractAntenna) -> None:
        chex.assert_trees_all_close(antenna.angular_frequency, 2 * jnp.pi * 1e9)

    def test_wavelength(self, antenna: AbstractAntenna) -> None:
        chex.assert_trees_all_close(antenna.wavelength, c / 1e9)

    def test_wavenumber(self, antenna: AbstractAntenna) -> None:
        chex.assert_trees_all_close(antenna.wavenumber, 2 * jnp.pi * 1e9 / c)

    def test_aperture(self, antenna: AbstractAntenna) -> None:
        chex.assert_trees_all_close(
            antenna.aperture, antenna.wavelength**2 / (4 * jnp.pi)
        )

    def test_abstract(self) -> None:
        with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class AbstractAntenna",
        ):
            _ = AbstractAntenna(frequency=jnp.asarray(1e9))

    @pytest.mark.parametrize("num_wavelengths", [None, 10.0])
    @pytest.mark.parametrize(
        ("backend", "expectation"),
        [
            pytest.param(
                "vispy",
                pytest.warns(
                    UserWarning,
                    match="VisPy does not currently support coloring like we would like",
                ),
                marks=skip_if_vispy_not_installed,
                id="vispy",
            ),
            pytest.param(
                "matplotlib",
                pytest.warns(
                    UserWarning,
                    match="Matplotlib requires 'colors' to be RGB or RGBA values",
                ),
                marks=skip_if_matplotlib_not_installed,
                id="matplotlib",
            ),
            pytest.param(
                "plotly",
                does_not_raise(),
                marks=skip_if_plotly_not_installed,
                id="plotly",
            ),
        ],
    )
    def test_plot_radiation_pattern(
        self,
        num_wavelengths: float | None,
        backend: str,
        expectation: AbstractContextManager[Exception],
        antenna: AbstractAntenna,
    ) -> None:
        with expectation:
            _ = antenna.plot_radiation_pattern(
                num_wavelengths=num_wavelengths, backend=backend
            )


class TestAbstractAntennaGenericEstimators:
    """Exercise the generic (unspecialized) estimators on :class:`AbstractAntenna`.

    :class:`Dipole` (and its subclasses) override :meth:`~AbstractAntenna.directivity`
    and :meth:`~AbstractAntenna.directive_gain` with closed-form expressions,
    so we use a minimal concrete antenna that does not, to cover the
    generic grid-based estimate implemented on :class:`AbstractAntenna` itself.
    """

    def test_directivity(self) -> None:
        antenna = _IsotropicAntenna(frequency=jnp.asarray(1e9))
        u, v, g = antenna.directivity(num_points=8)
        assert u.shape == (16,)
        assert v.shape == (8,)
        assert g.shape == (16, 8)
        # The fields (and thus the Poynting vector norm) are the same in
        # every direction, so the resulting directivity estimate should be
        # constant across the whole grid.
        chex.assert_trees_all_close(g, jnp.full_like(g, g[0, 0]))

    def test_directive_gain(self) -> None:
        antenna = _IsotropicAntenna(frequency=jnp.asarray(1e9))
        gain = antenna.directive_gain(num_points=8)
        assert gain.shape == ()
        chex.assert_trees_all_close(gain, antenna.directivity(num_points=8)[-1].max())


class TestDipole:
    def test_init(self) -> None:
        dipole = Dipole(
            1e9,
            current=2.0,
            length=4.0,
        )
        chex.assert_trees_all_close(
            jnp.linalg.norm(dipole.moment), (2.0 * 4.0 / dipole.angular_frequency)
        )
        dipole = Dipole(
            1e9,
            current=None,
        )
        chex.assert_trees_all_close(jnp.linalg.norm(dipole.moment), 1.0)
        dipole = Dipole(1e9, charge=3.0, length=2.0)
        chex.assert_trees_all_close(
            jnp.linalg.norm(dipole.moment),
            3.0 * 2.0,
        )

    def test_look_at(self) -> None:
        dipole = Dipole(1e9)
        got = normalize(dipole.moment)[0]
        expected = jnp.array([0.0, 0.0, +1.0])
        chex.assert_trees_all_equal(got, expected)

        dipole = Dipole(1e9, look_at=jnp.array([0.0, 0.0, -1.0]))
        got = normalize(dipole.moment)[0]
        expected = jnp.array([1.0, 0.0, 0.0])
        chex.assert_trees_all_close(got, expected, atol=1e-6)

        dipole = Dipole(1e9, look_at=jnp.array([1.0, 1.0, 0.0]))
        got = normalize(dipole.moment)[0]
        expected = jnp.array([0.0, 0.0, -1.0])
        chex.assert_trees_all_close(got, expected, atol=1e-6)

        dipole = Dipole(1e9, look_at=jnp.array([1.0, 0.0, -1.0]))
        got = normalize(dipole.moment)[0]
        expected = jnp.array([-1.0, 0.0, -1.0]) / jnp.sqrt(2.0)
        chex.assert_trees_all_close(got, expected, atol=1e-6)

    @pytest.mark.parametrize("frequency", [0.1e9, 1e9, 10e9])
    def test_reference_power(self, frequency: float, key: PRNGKeyArray) -> None:
        key_pa, key_moment = jax.random.split(key, 2)
        xyz = spherical_to_cartesian(
            jax.random.uniform(key_pa, (10_000, 2), maxval=jnp.pi)
        )
        dipole = Dipole(
            frequency=frequency,
            moment=normalize(jax.random.normal(key_moment, (3,)))[0],
        )
        expected = (
            jnp.linalg.norm(dipole.poynting_vector(xyz), axis=-1).max() * 4 * jnp.pi
        )
        chex.assert_trees_all_close(dipole.reference_power, expected, rtol=1e-2)

    @pytest.mark.parametrize(
        ("ratio", "expected_gain"),
        [(0.5, 1.5), (1.0, 1.5), (1.25, 1.5), (1.5, 1.5), (2.0, 1.5)],
    )
    def test_directivity(self, ratio: float, expected_gain: float) -> None:
        f = 1e9
        dipole = Dipole(
            frequency=f,
            num_wavelengths=ratio,
        )
        directive_gain = dipole.directive_gain(1000)
        chex.assert_trees_all_close(directive_gain, expected_gain)

    def test_directivity_pattern(self) -> None:
        dipole = Dipole(frequency=1e9)
        u, v, g = dipole.directivity(num_points=8)
        assert u.shape == (16,)
        assert v.shape == (8,)
        assert g.shape == (16, 8)

        # Closed-form directivity of a Hertzian dipole: D(theta) = 1.5 * sin(theta)**2,
        # where theta is the angle from the dipole moment (here, the z-axis).
        expected = 1.5 * jnp.sin(v) ** 2
        chex.assert_trees_all_close(g, jnp.broadcast_to(expected, g.shape), atol=1e-6)

        # Directivity vanishes along the dipole axis (theta = 0).
        chex.assert_trees_all_close(g[:, 0], jnp.zeros_like(g[:, 0]), atol=1e-6)

    def test_wavefront_radii(self) -> None:
        # A dipole is an ideal point source, with zero offset from its
        # own 'center' in every direction (the default 'AbstractAntenna'
        # behavior, which 'Dipole' does not override).
        dipole = Dipole(frequency=1e9, center=jnp.array([1.0, 2.0, 3.0]))
        chex.assert_trees_all_close(
            dipole.wavefront_radii(jnp.array([1.0, 0.0, 0.0])), 0.0
        )
        chex.assert_trees_all_close(
            dipole.wavefront_radii(jnp.array([0.0, 0.0, 1.0])), 0.0
        )

        # Batched directions.
        k_hat = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        chex.assert_trees_all_close(dipole.wavefront_radii(k_hat), jnp.zeros(2))


class TestShortDipole:
    @pytest.mark.skip
    @pytest.mark.parametrize(
        ("ratio", "expected_gain_dbi"),
        [(0.5, 2.15), (1.0, 4.0), (1.25, 5.2), (1.5, 3.5), (2.0, 4.3)],
    )
    def test_directivity(self, ratio: float, expected_gain_dbi: float) -> None:
        pass

    def test_fields_not_implemented(self) -> None:
        dipole = ShortDipole(frequency=1e9)
        with pytest.raises(NotImplementedError):
            dipole.fields(jnp.array([1.0, 0.0, 0.0]))

    def test_directivity_bypasses_dipole(self) -> None:
        # 'ShortDipole' explicitly bypasses 'Dipole's specialized
        # 'directivity'/'directive_gain' by calling back into
        # 'AbstractAntenna's generic implementation, which in turn needs
        # 'fields' (not implemented for 'ShortDipole' yet), so both raise.
        dipole = ShortDipole(frequency=1e9)
        with pytest.raises(NotImplementedError):
            dipole.directivity()
        with pytest.raises(NotImplementedError):
            dipole.directive_gain()


class TestFarFieldDipoleAntenna:
    def test_wavefront_radii_is_always_none(self) -> None:
        # A 'FarFieldDipoleAntenna' unconditionally reports a planar
        # wavefront, regardless of direction.
        antenna = FarFieldDipoleAntenna(
            frequency=1e9, center=jnp.array([1.0, 2.0, 3.0])
        )
        assert antenna.wavefront_radii(jnp.array([1.0, 0.0, 0.0])) is None
        assert antenna.wavefront_radii(jnp.array([0.0, 1.0, 0.0])) is None

    def test_inherits_dipole_behavior(self) -> None:
        # Everything else (fields, reference_power, directivity, ...) is
        # inherited from 'Dipole' unchanged.
        frequency = 1e9
        dipole = Dipole(frequency=frequency, num_wavelengths=0.5)
        far_field = FarFieldDipoleAntenna(frequency=frequency, num_wavelengths=0.5)

        assert isinstance(far_field, Dipole)
        assert isinstance(far_field, AbstractFarFieldAntenna)
        assert isinstance(far_field, AbstractAntenna)

        chex.assert_trees_all_close(far_field.moment, dipole.moment)
        chex.assert_trees_all_close(far_field.reference_power, dipole.reference_power)

        r = jnp.array([[10.0, 0.0, 0.0], [0.0, 5.0, 3.0]])
        e_dipole, b_dipole = dipole.fields(r)
        e_far_field, b_far_field = far_field.fields(r)
        chex.assert_trees_all_close(e_dipole, e_far_field)
        chex.assert_trees_all_close(b_dipole, b_far_field)

    def test_abstract_base_class(self) -> None:
        # 'AbstractFarFieldAntenna' itself still needs 'fields' and
        # 'reference_power' from a subclass, just like a plain 'AbstractAntenna'.
        with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class AbstractFarFieldAntenna",
        ):
            _ = AbstractFarFieldAntenna(frequency=jnp.asarray(1e9))


class TestAbstractRadiationPattern:
    def test_abstract(self) -> None:
        with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class AbstractRadiationPattern",
        ):
            _ = AbstractRadiationPattern(frequency=jnp.asarray(1e9))

    def test_directivity(self) -> None:
        pattern = _ConstantPolarizationPattern(frequency=jnp.asarray(1e9))
        u, v, g = pattern.directivity(num_points=8)
        assert u.shape == (16,)
        assert v.shape == (8,)
        assert g.shape == (16, 8)
        # s = r_hat * [1, 0, 0] and p = r_hat * [0, 1, 0], so
        # ||s||^2 + ||p||^2 = r_hat_x^2 + r_hat_y^2 = sin(v)^2 (independent
        # of the azimuthal angle u).
        expected = jnp.broadcast_to(jnp.sin(v) ** 2, g.shape)
        chex.assert_trees_all_close(g, expected, atol=1e-6)

    def test_directive_gain(self) -> None:
        pattern = _ConstantPolarizationPattern(frequency=jnp.asarray(1e9))
        gain = pattern.directive_gain(num_points=8)
        assert gain.shape == ()
        chex.assert_trees_all_close(gain, pattern.directivity(num_points=8)[-1].max())

    @pytest.mark.parametrize("num_wavelengths", [None, 10.0])
    @pytest.mark.parametrize(
        ("backend", "expectation"),
        [
            pytest.param(
                "vispy",
                pytest.warns(
                    UserWarning,
                    match="VisPy does not currently support coloring like we would like",
                ),
                marks=skip_if_vispy_not_installed,
                id="vispy",
            ),
            pytest.param(
                "matplotlib",
                pytest.warns(
                    UserWarning,
                    match="Matplotlib requires 'colors' to be RGB or RGBA values",
                ),
                marks=skip_if_matplotlib_not_installed,
                id="matplotlib",
            ),
            pytest.param(
                "plotly",
                does_not_raise(),
                marks=skip_if_plotly_not_installed,
                id="plotly",
            ),
        ],
    )
    def test_plot_radiation_pattern(
        self,
        num_wavelengths: float | None,
        backend: str,
        expectation: AbstractContextManager[Exception],
    ) -> None:
        pattern = _ConstantPolarizationPattern(frequency=jnp.asarray(1e9))
        with expectation:
            _ = pattern.plot_radiation_pattern(
                num_wavelengths=num_wavelengths, backend=backend
            )


class TestHWDipolePattern:
    def test_polarization_vectors_is_unfinished(self) -> None:
        # 'HWDipolePattern.polarization_vectors' is not implemented yet.
        pattern = HWDipolePattern(
            frequency=jnp.asarray(1e9), direction=jnp.array([0.0, 0.0, 1.0])
        )
        with pytest.raises(NotImplementedError):
            pattern.polarization_vectors(jnp.array([1.0, 0.0, 0.0]))
