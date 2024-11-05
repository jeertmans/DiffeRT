import pytest
import scipy.constants

from differt.em import constants


@pytest.mark.parametrize(
    ("constant_name", "scipy_name"),
    [
        ("c", None),
        ("epsilon_0", None),
        ("mu_0", None),
        ("z_0", "characteristic_impedance_of_vacuum"),
    ],
)
def test_constants(constant_name: str, scipy_name: str | None) -> None:
    got = getattr(constants, constant_name)
    if scipy_name:
        expected = getattr(scipy.constants, scipy_name)
    else:
        expected = getattr(scipy.constants, constant_name)
    assert got == expected
