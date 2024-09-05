from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from differt.plotting import draw_image, draw_markers, draw_mesh, draw_paths, use


@pytest.mark.parametrize(
    "backend",
    ["vispy", "matplotlib", "plotly"],
)
def test_draw_mesh(
    backend: str,
) -> None:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    with use(backend):
        _ = draw_mesh(vertices, triangles)


@pytest.mark.parametrize(
    "backend",
    ["vispy", "matplotlib", "plotly"],
)
def test_draw_paths(
    rng: np.random.Generator,
    backend: str,
) -> None:
    paths = rng.random(size=(10, 4, 3))
    with use(backend):
        _ = draw_paths(paths)


@pytest.mark.parametrize(
    ("backend", "expectation"),
    [
        ("vispy", does_not_raise()),
        ("matplotlib", pytest.raises(NotImplementedError)),
        ("plotly", does_not_raise()),
    ],
)
@pytest.mark.parametrize("with_labels", [True, False])
def test_draw_markers(
    rng: np.random.Generator,
    backend: str,
    expectation: AbstractContextManager[Exception],
    with_labels: bool,
) -> None:
    markers = rng.random(size=(4, 3))

    labels = ["A", "B", "C", "D"] if with_labels else None

    with use(backend), expectation:
        _ = draw_markers(markers, labels=labels)


@pytest.mark.parametrize(
    "backend",
    ["vispy", "matplotlib", "plotly"],
)
@pytest.mark.parametrize(
    "pass_xy",
    [True, False],
)
def test_draw_image(
    backend: str,
    pass_xy: bool,
) -> None:
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x, y)  # noqa: N806
    data = X * Y

    with use(backend):
        _ = draw_image(data, x=x if pass_xy else None, y=y if pass_xy else None)
