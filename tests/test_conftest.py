import pytest


def test_conftest_is_excluded() -> None:
    with pytest.raises(ImportError):
        from differt.conftest import add_doctest_modules  # noqa: F401
