from pathlib import Path

import jax.numpy as jnp
import pytest

from differt.geometry._triangle_mesh import TriangleMesh


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
def two_buildings_mesh(two_buildings_obj_file: str) -> TriangleMesh:
    return TriangleMesh.load_obj(two_buildings_obj_file)


@pytest.fixture(scope="session")
def sphere_mesh() -> TriangleMesh:
    vsp_geom = pytest.importorskip("vispy.geometry", reason="vispy not installed")

    mesh = vsp_geom.create_sphere()

    vertices = jnp.asarray(mesh.get_vertices())
    triangles = jnp.asarray(mesh.get_faces(), dtype=jnp.int32)
    return TriangleMesh(vertices=vertices, triangles=triangles)
