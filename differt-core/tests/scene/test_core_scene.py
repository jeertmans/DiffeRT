import pytest


def test_triangle_scene_deprecated() -> None:
    from differt_core.scene import TriangleScene  # noqa: PLC0415

    with pytest.warns(DeprecationWarning, match="TriangleScene is deprecated"):
        with pytest.raises(TypeError):
            TriangleScene()
