from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import chex
import jax.numpy as jnp
import pytest

from differt.em import c, mu_0
from differt.em._antenna import Antenna, Dipole, ShortDipole


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

    def test_abstract(self):
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

    def test_average_power(self) -> None:
        f = 1e9
        w = 2 * jnp.pi * f
        p_0 = 1.0
        dipole = Dipole(
            frequency=f,
        )
        p_0 = jnp.linalg.norm(dipole.moment)
        chex.assert_trees_all_close(
            dipole.average_power, mu_0 * w**4 * p_0**2 / (12 * jnp.pi * c)
        )

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
        f = 1e9
        dipole = ShortDipole(
            frequency=f,
            num_wavelengths=ratio,
        )
        directive_gain = dipole.directive_gain(1000)
        directive_gain_dbi = 10 * jnp.log10(directive_gain)
        chex.assert_trees_all_close(directive_gain_dbi, expected_gain_dbi)
