from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import pytest

from differt.geometry._mesh import Mesh
from differt.scene._scene import Scene
from differt.scene._sionna import (
    SIONNA_SCENES_FOLDER,
    download_sionna_scenes,
    get_sionna_scene,
)


@pytest.fixture(scope="session")
def sionna_folder() -> Path:
    download_sionna_scenes(folder=SIONNA_SCENES_FOLDER)
    return SIONNA_SCENES_FOLDER


@pytest.fixture(scope="module")
def advanced_path_tracing_example_scene(
    two_buildings_mesh: Mesh,
) -> Scene:
    tx = jnp.array([0.0, 4.9352, 22.0])
    rx = jnp.array([0.0, 10.034, 1.50])

    return Scene(transmitters=tx, receivers=rx, mesh=two_buildings_mesh)


@pytest.fixture(scope="module")
def simple_street_canyon_scene(sionna_folder: Path) -> Scene:
    file = get_sionna_scene("simple_street_canyon", folder=sionna_folder)
    scene = Scene.load_xml(file)
    scene = eqx.tree_at(lambda s: s.transmitters, scene, jnp.array([-22.0, 0.0, 32.0]))
    return eqx.tree_at(lambda s: s.receivers, scene, jnp.array([+22.0, 0.0, 32.0]))
