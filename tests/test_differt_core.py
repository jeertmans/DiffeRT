from pathlib import Path

import differt_core
import tomli

import differt


def test_same_version() -> None:
    assert differt.__version__ == differt_core.__version__


def test_version_matches_cargo_toml(differt_core_cargo_toml: Path) -> None:
    toml = tomli.loads(differt_core_cargo_toml.read_text())
    assert differt_core.__version__ == toml["package"]["version"]