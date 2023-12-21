import differt


def test_same_version() -> None:
    assert differt.__version__ == differt._core.__version__
