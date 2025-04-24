from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PRNGKeyArray

from differt.em import c
from differt.em._antenna import (
    Antenna,
    Dipole,
)
from differt.geometry import normalize, spherical_to_cartesian


@pytest.fixture
def antenna() -> Dipole:
    return Dipole(
        frequency=1e9,
    )


class TestAntenna:
    def test_frequency(self, antenna: Antenna) -> None:
        chex.assert_trees_all_equal(antenna.frequency, 1e9)

    def test_center(self, antenna: Antenna) -> None:
        chex.assert_trees_all_equal(antenna.center, jnp.zeros(3))

    def test_period(self, antenna: Antenna) -> None:
        chex.assert_trees_all_close(antenna.period, 1 / 1e9)

    def test_angular_frequency(self, antenna: Antenna) -> None:
        chex.assert_trees_all_close(antenna.angular_frequency, 2 * jnp.pi * 1e9)

    def test_wavelength(self, antenna: Antenna) -> None:
        chex.assert_trees_all_close(antenna.wavelength, c / 1e9)

    def test_wavenumber(self, antenna: Antenna) -> None:
        chex.assert_trees_all_close(antenna.wavenumber, 2 * jnp.pi * 1e9 / c)

    def test_abstract(self) -> None:
        with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class Antenna",
        ):
            _ = Antenna(frequency=1e9)  # type: ignore[reportAbstractUsage]

    @pytest.mark.parametrize("num_wavelengths", [None, 10.0])
    @pytest.mark.parametrize(
        ("backend", "expectation"),
        [
            (
                "vispy",
                pytest.warns(
                    UserWarning,
                    match="VisPy does not currently support coloring like we would like",
                ),
            ),
            (
                "matplotlib",
                pytest.warns(
                    UserWarning,
                    match="Matplotlib requires 'colors' to be RGB or RGBA values",
                ),
            ),
            ("plotly", does_not_raise()),
        ],
    )
    def test_plot_radiation_pattern(
        self,
        num_wavelengths: float | None,
        backend: str,
        expectation: AbstractContextManager[Exception],
        antenna: Antenna,
    ) -> None:
        with expectation:
            _ = antenna.plot_radiation_pattern(
                num_wavelengths=num_wavelengths, backend=backend
            )


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


class TestShortDipole:
    @pytest.mark.skip
    @pytest.mark.parametrize(
        ("ratio", "expected_gain_dbi"),
        [(0.5, 2.15), (1.0, 4.0), (1.25, 5.2), (1.5, 3.5), (2.0, 4.3)],
    )
    def test_directivity(self, ratio: float, expected_gain_dbi: float) -> None:
        pass
