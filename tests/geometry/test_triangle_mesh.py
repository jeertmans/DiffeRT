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
            [0.1, 0.8, 0.0],  # Inside
            [0.8, 0.1, 0.0],  # Inside
            [0.1, 0.1, 0.0],  # Inside
            [0.3, 0.3, 0.0],  # Inside
            [1.0, 1.0, 0.0],  # Outside
            [0.3, 1.3, 0.0],  # Outside
            [0.0, 0.0, 0.0],  # Inside but on vertex
            [1.0, 0.0, 0.0],  # Inside but on vertex
            [0.0, 1.0, 0.0],  # Inside but on vertex
            [0.5, 0.0, 0.0],  # Inside but on edge
            [0.0, 0.5, 0.0],  # Inside but on edge
            [0.5, 0.5, 0.0],  # Inside but on edge
        ]
    )
    expected = jnp.array(
        [True, True, True, True, False, False, True, True, True, True, True, True]
    )
    n = vertices.shape[0]
    triangle_vertices = jnp.tile(triangle_vertices, (n, 1, 1))
    got = triangles_contain_vertices_assuming_inside_same_plane(
        triangle_vertices, vertices
    )
    chex.assert_trees_all_equal(got, expected)


class TestTriangleMesh:
    def test_load_obj(self, two_buildings_obj_file: Path) -> None:
        mesh = TriangleMesh.load_obj(two_buildings_obj_file)
        assert len(mesh._mesh.triangles) == 24
        
    def test_normals(self, two_buildings_mesh: TriangleMesh) -> None:
        chex.assert_equal_shape((two_buildings_mesh.normals, two_buildings_mesh.triangles))
        got = jnp.linalg.norm(two_buildings_mesh.normals, axis=-1)
        expected = jnp.ones_like(got)
        chex.assert_trees_all_close(got, expected)
