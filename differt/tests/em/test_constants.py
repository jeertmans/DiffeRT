import pytest
import scipy.constants

from differt.em import constants


@pytest.mark.parametrize(
    ("constant_name", "value"),
    [
        ("c", None),
        ("epsilon_0", None),
        ("mu_0", None),
        (
            "z_0",
            scipy.constants.physical_constants["characteristic impedance of vacuum"][0],
        ),
    ],
)
def test_constants(constant_name: str, value: float | None) -> None:
    got = getattr(constants, constant_name)
    if value:
        assert abs(got - value) < 1e-6
    else:
        assert got == getattr(scipy.constants, constant_name)
