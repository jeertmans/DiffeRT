import differt
from differt import _core


def test_same_version() -> None:
    assert differt.__version__ == _core.__version__
