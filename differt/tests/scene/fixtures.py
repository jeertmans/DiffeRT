from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import pytest

from differt.geometry.triangle_mesh import TriangleMesh
from differt.scene.sionna import (
    SIONNA_SCENES_FOLDER,
    download_sionna_scenes,
    get_sionna_scene,
)
from differt.scene.triangle_scene import TriangleScene


@pytest.fixture(scope="session")
def sionna_folder() -> Path:
    download_sionna_scenes(folder=SIONNA_SCENES_FOLDER)
    return SIONNA_SCENES_FOLDER


@pytest.fixture(scope="module")
def advanced_path_tracing_example_scene(
    two_buildings_mesh: TriangleMesh,
) -> TriangleScene:
    tx = jnp.array([0.0, 4.9352, 22.0])
    rx = jnp.array([0.0, 10.034, 1.50])

    return TriangleScene(transmitters=tx, receivers=rx, mesh=two_buildings_mesh)


@pytest.fixture(scope="module")
def simple_street_canyon_scene(sionna_folder: Path) -> TriangleScene:
    file = get_sionna_scene("simple_street_canyon", folder=sionna_folder)
    scene = TriangleScene.load_xml(file)
    scene = eqx.tree_at(lambda s: s.transmitters, scene, jnp.array([-37.0, 14.0, 35.0]))
    return eqx.tree_at(lambda s: s.receivers, scene, jnp.array([12.0, 0.0, 35.0]))
