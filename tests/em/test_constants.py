import pytest
import scipy.constants

from differt.em import constants


@pytest.mark.parametrize("constant_name", constants.__all__)
def test_constants(constant_name: str):
    got = getattr(constants, constant_name)
    expected = getattr(scipy.constants, constant_name)
    assert got == expected
