import pytest


def test_triangle_mesh_deprecated() -> None:
    from differt_core.geometry import (  # ruff:ignore[import-outside-top-level]
        TriangleMesh,
    )

    with pytest.warns(DeprecationWarning, match="TriangleMesh is deprecated"):
        with pytest.raises(TypeError):
            TriangleMesh()
