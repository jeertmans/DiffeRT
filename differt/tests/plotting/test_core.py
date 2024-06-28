import numpy as np
import pytest

from differt.plotting import draw_mesh, use


@pytest.mark.parametrize(
    "backend",
    ("vispy", "matplotlib", "plotly"),
)
def test_draw_mesh(
    backend: str,
) -> None:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    with use(backend):
        _ = draw_mesh(vertices, triangles)
