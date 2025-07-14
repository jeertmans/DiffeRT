import differt
import differt_core


def test_same_version() -> None:
    assert differt.__version__ == differt_core.__version__
    assert differt.__version_info__ == differt_core.__version_info__
