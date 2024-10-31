import pytest
import scipy.constants

from differt.em import _constants


@pytest.mark.parametrize("constant_name", ["c", "epsilon_0", "mu_0"])
def test_constants(constant_name: str) -> None:
    got = getattr(_constants, constant_name)
    expected = getattr(scipy.constants, constant_name)
    assert got == expected
