import differt_core


def test_version() -> None:
    assert (
        tuple(int(i) for i in differt_core.__version__.split(".")[:3])
        == differt_core.__version_info__
    )
