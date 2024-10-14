from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from differt.plotting import (
    draw_contour,
    draw_image,
    draw_markers,
    draw_mesh,
    draw_paths,
    draw_rays,
    draw_surface,
    use,
)


@pytest.mark.parametrize(
    "backend",
    ["vispy", "matplotlib", "plotly"],
)
def test_draw_mesh(
    backend: str,
) -> None:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    with use(backend):
        _ = draw_mesh(vertices, triangles)


@pytest.mark.parametrize(
    "backend",
    ["vispy", "matplotlib", "plotly"],
)
def test_draw_paths(
    backend: str,
    rng: np.random.Generator,
) -> None:
    paths = rng.random(size=(10, 4, 3))
    with use(backend):
        _ = draw_paths(paths)


@pytest.mark.parametrize(
    "backend",
    ["vispy", "matplotlib", "plotly"],
)
def test_draw_rays(
    backend: str,
    rng: np.random.Generator,
) -> None:
    ray_origins, ray_directions = rng.random(size=(2, 10, 4, 3))
    with use(backend):
        _ = draw_rays(ray_origins, ray_directions)


@pytest.mark.parametrize(
    "backend",
    ["vispy", "matplotlib", "plotly"],
)
@pytest.mark.parametrize("with_labels", [True, False])
def test_draw_markers(
    backend: str,
    with_labels: bool,
    rng: np.random.Generator,
) -> None:
    markers = rng.random(size=(4, 3))

    labels = ["A", "B", "C", "D"] if with_labels else None

    if backend == "matplotlib" and with_labels:
        expectation = pytest.warns(
            UserWarning,
            match="Matplotlib does not currently support adding labels to markers",
        )
    else:
        expectation = does_not_raise()

    with use(backend), expectation:
        _ = draw_markers(markers, labels=labels, text_kwargs={"rotation": 30})


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
    # VisPY does not support float64 and will complain if provided
    x = np.linspace(0, 1, 10, dtype=np.float32)
    y = np.linspace(0, 1, 20, dtype=np.float32)
    X, Y = np.meshgrid(x, y)  # noqa: N806
    data = X * Y

    with use(backend):
        _ = draw_image(data, x=x if pass_xy else None, y=y if pass_xy else None)


@pytest.mark.parametrize(
    "backend",
    ["vispy", "matplotlib", "plotly"],
)
@pytest.mark.parametrize(
    "pass_xy",
    [True, False],
)
@pytest.mark.parametrize("fill", [False, True])
@pytest.mark.parametrize("levels", [None, 10, np.linspace(0, 1, 21)])
def test_draw_contour(
    backend: str, pass_xy: bool, fill: bool, levels: int | np.ndarray | None
) -> None:
    # VisPY does not support float64 and will complain if provided
    x = np.linspace(0, 1, 10, dtype=np.float32)
    y = np.linspace(0, 1, 20, dtype=np.float32)
    X, Y = np.meshgrid(x, y)  # noqa: N806
    data = X * Y

    if (backend == "vispy" and (fill or isinstance(levels, int))) or (
        backend == "plotly" and isinstance(levels, np.ndarray)
    ):
        expectation = pytest.warns(UserWarning)
    else:
        expectation = does_not_raise()

    with use(backend), expectation:
        _ = draw_contour(
            data,
            x=x if pass_xy else None,
            y=y if pass_xy else None,
            levels=levels,
            fill=fill,
        )


@pytest.mark.parametrize(
    "backend",
    ["vispy", "matplotlib", "plotly"],
)
@pytest.mark.parametrize(
    "pass_xy",
    [True, False],
)
@pytest.mark.parametrize(
    "pass_colors",
    [True, False],
)
def test_draw_surface(
    backend: str,
    pass_xy: bool,
    pass_colors: bool,
) -> None:
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x, y)  # noqa: N806
    Z = X * Y

    with use(backend):
        _ = draw_surface(
            x=x if pass_xy else None,
            y=y if pass_xy else None,
            z=z,
            colors=X * X + Y * Y + Z * Z if pass_colors else None,
        )
