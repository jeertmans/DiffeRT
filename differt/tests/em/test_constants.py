import pytest
import scipy.constants

from differt.em import _constants


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
    got = getattr(_constants, constant_name)
    expected: float = (
        value if value is not None else getattr(scipy.constants, constant_name)
    )

    assert abs(got - expected) < 1e-6
