from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import pytest

from differt.geometry._mesh import Mesh
from differt.geometry._scene import Scene
from differt.geometry._sionna import (
    SIONNA_SCENES_FOLDER,
    download_sionna_scenes,
    get_sionna_scene,
)

from .utils import PlanarMirrorsSetup


@pytest.fixture(scope="session")
def two_buildings_obj_file() -> str:
    return (
        Path(__file__)
        .parent.joinpath("two_buildings.obj")
        .resolve(strict=True)
        .as_posix()
    )


@pytest.fixture(scope="session")
def two_buildings_obj_with_mat_file() -> str:
    return (
        Path(__file__)
        .parent.joinpath("two_buildings_with_mat.obj")
        .resolve(strict=True)
        .as_posix()
    )


@pytest.fixture(scope="session")
def two_buildings_ply_file() -> str:
    return (
        Path(__file__)
        .parent.joinpath("two_buildings.ply")
        .resolve(strict=True)
        .as_posix()
    )


@pytest.fixture(scope="session")
def cube_ply_file() -> str:
    return Path(__file__).parent.joinpath("cube.ply").resolve(strict=True).as_posix()


@pytest.fixture(scope="session")
def two_buildings_mesh(two_buildings_obj_file: str) -> Mesh:
    return Mesh.load_obj(two_buildings_obj_file)


@pytest.fixture(scope="session")
def sphere_mesh() -> Mesh:
    vsp_geom = pytest.importorskip("vispy.geometry", reason="vispy not installed")

    mesh = vsp_geom.create_sphere()

    vertices = jnp.asarray(mesh.get_vertices())
    triangles = jnp.asarray(mesh.get_faces(), dtype=jnp.int32)
    return Mesh(vertices=vertices, triangles=triangles)


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


@pytest.fixture(scope="session")
def basic_planar_mirrors_setup() -> PlanarMirrorsSetup:
    """
    Test setup that looks something like:

                1           3
             ───────     ───────
           0                           5
    (from) x                           x (to)

                   ───────      ───────
                      2            4

    where xs are starting and ending vertices, and '───────' are mirrors.
    """
    return PlanarMirrorsSetup(
        from_vertices=jnp.array([0.0, 0.0, 0.0]),
        to_vertices=jnp.array([1.0, 0.0, 0.0]),
        mirror_vertices=jnp.array([
            [0.0, +1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, +1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]),
        mirror_normals=jnp.array(
            [[0.0, -1.0, 0.0], [0.0, +1.0, 0.0], [0.0, -1.0, 0.0], [0.0, +1.0, 0.0]],
        ),
        paths=jnp.array(
            [
                [1.0 / 8.0, +1.0, 0.0],
                [3.0 / 8.0, -1.0, 0.0],
                [5.0 / 8.0, +1.0, 0.0],
                [7.0 / 8.0, -1.0, 0.0],
            ],
        ),
    )
