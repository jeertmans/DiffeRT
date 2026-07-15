import pytest


def test_triangle_mesh_deprecated() -> None:
    from differt_core.geometry import TriangleMesh  # noqa: PLC0415

    with pytest.warns(DeprecationWarning, match="TriangleMesh is deprecated"):
        with pytest.raises(TypeError):
            TriangleMesh()
