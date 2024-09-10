from pathlib import Path

import pytest

from differt.scene.sionna import (
    SIONNA_SCENES_FOLDER,
    download_sionna_scenes,
    get_sionna_scene,
    list_sionna_scenes,
)
from differt_core.scene.sionna import SionnaScene


@pytest.fixture
def folder() -> Path:
    return SIONNA_SCENES_FOLDER


@pytest.fixture
def empty_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("scenes")


def test_download_sionna_scenes(folder: Path) -> None:
    download_sionna_scenes(folder=folder)
    download_sionna_scenes(folder=str(folder))


def test_download_sionna_scenes_existing_empty_folder(empty_folder: Path) -> None:
    download_sionna_scenes(folder=empty_folder, cached=False)


def test_download_sionna_scenes_existing_non_empty_folder(folder: Path) -> None:
    with pytest.raises(OSError, match="[Dd]irectory (is )?not empty"):
        download_sionna_scenes(folder=folder, cached=False)


def test_list_sionna_scenes(folder: Path) -> None:
    list_sionna_scenes(folder=folder)
    list_sionna_scenes(folder=str(folder))


@pytest.mark.parametrize("scene_name", ["foo", "bar"])
def test_get_unexisting_sionna_scene(scene_name: str, folder: Path) -> None:
    with pytest.raises(ValueError, match="Cannot find scene_name"):
        _ = get_sionna_scene(scene_name, folder=folder)


@pytest.mark.parametrize("scene_name", ["box", "etoile", "munich"])
def test_get_existing_sionna_scene(scene_name: str, folder: Path) -> None:
    assert Path(get_sionna_scene(scene_name, folder=folder)).exists()
    assert Path(get_sionna_scene(scene_name, folder=str(folder))).exists()


class TestSionnaScene:
    def test_load_xml(self, folder: Path) -> None:
        for scene_name in list_sionna_scenes(folder=folder):
            file = get_sionna_scene(scene_name, folder=folder)
            scene = SionnaScene.load_xml(file)

            assert len(scene.shapes) > 0
            assert len(scene.materials) > 0

            for shape in scene.shapes.values():
                assert shape.material_id in scene.materials
