import pytest


def test_triangle_scene_deprecated() -> None:
    from differt_core.geometry import (  # ruff:ignore[import-outside-top-level]
        TriangleScene,
    )

    with pytest.warns(DeprecationWarning, match="TriangleScene is deprecated"):
        with pytest.raises(TypeError):
            TriangleScene()
