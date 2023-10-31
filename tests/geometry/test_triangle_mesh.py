import pytest
from typing import Iterator

from differt.geometry import TriangleMesh

from pathlib import Path


@pytest.fixture(scope="module")
def two_buildings_obj_file() -> Iterator[Path]:
    yield Path(__file__).parent.joinpath("two_buildings.obj").resolve(strict=True)


@pytest.fixture(scope="module")
def two_buildings_mesh(two_buildings_obj_file: Path) -> Iterator[TriangleMesh]:
    yield TriangleMesh.load_obj(two_buildings_obj_file)


class TestTriangleMesh:
    def test_load_obj(self, two_buildings_obj_file: Path) -> None:
        mesh = TriangleMesh.load_obj(two_buildings_obj_file)
        assert len(mesh.mesh.triangles) == 24
