import pytest

import differt
import differt.em
import differt.geometry


@pytest.mark.parametrize(
    ("name", "submodule"),
    [
        ("Scene", differt.geometry),
        ("Mesh", differt.geometry),
        ("TracedPaths", differt.geometry),
        ("LaunchedPaths", differt.geometry),
        ("Material", differt.em),
        ("InteractionType", differt.em),
        ("SpecularReflection", differt.em),
        ("Diffraction", differt.em),
        ("Scattering", differt.em),
        ("Transmission", differt.em),
        ("RIS", differt.em),
        ("WavefrontState", differt.em),
        ("propagate_wavefront", differt.em),
    ],
)
def test_lazy_top_level_reexport_matches_submodule(
    name: str, submodule: object
) -> None:
    assert getattr(differt, name) is getattr(submodule, name)


def test_lazy_top_level_reexport_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError, match="unknown_attribute"):
        _ = differt.unknown_attribute


def test_lazy_top_level_reexport_appears_in_dir() -> None:
    assert "Scene" in dir(differt)
    assert "SpecularReflection" in dir(differt)
