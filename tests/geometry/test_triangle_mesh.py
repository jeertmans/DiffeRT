from collections.abc import Iterator
from pathlib import Path

import chex
import jax.numpy as jnp
import pytest

from differt.geometry.triangle_mesh import (
    TriangleMesh,
    triangles_contain_vertices_assuming_inside_same_plane,
)


@pytest.fixture(scope="module")
def two_buildings_obj_file() -> Iterator[Path]:
    yield Path(__file__).parent.joinpath("two_buildings.obj").resolve(strict=True)


@pytest.fixture(scope="module")
def two_buildings_mesh(two_buildings_obj_file: Path) -> Iterator[TriangleMesh]:
    yield TriangleMesh.load_obj(two_buildings_obj_file)


def test_triangles_contain_vertices_assuming_inside_same_planes() -> None:
    triangle_vertices = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    vertices = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.3, 0.3, 0.0],
            [1.0, 1.0, 0.0],
            [0.3, 0.3, 1.0],
        ]
    )
    expected = jnp.array([True, True, True, True, False, False])
    n = vertices.shape[0]
    triangle_vertices = jnp.tile(triangle_vertices, (n, 1, 1))
    got = triangles_contain_vertices_assuming_inside_same_plane(
        triangle_vertices, vertices
    )
    chex.assert_trees_all_equal(got, expected)


class TestTriangleMesh:
    def test_load_obj(self, two_buildings_obj_file: Path) -> None:
        mesh = TriangleMesh.load_obj(two_buildings_obj_file)
        assert len(mesh.mesh.triangles) == 24
