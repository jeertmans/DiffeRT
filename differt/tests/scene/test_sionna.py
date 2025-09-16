from functools import partial
from pathlib import Path
from timeit import timeit

import pytest
from pytest_subtests import SubTests

from differt.scene._sionna import (
    download_sionna_scenes as no_timeout_download_sionna_scenes,
)
from differt.scene._sionna import (
    get_sionna_scene,
    list_sionna_scenes,
)
from differt_core.scene import SionnaScene

# Let's put a timeout on downloading the scenes.
download_sionna_scenes = partial(no_timeout_download_sionna_scenes, timeout=600.0)


@pytest.fixture
def empty_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("scenes")


def test_download_sionna_scenes_cached(sionna_folder: Path) -> None:
    # Downloading should be pretty fast, because we use cached folder
    assert timeit(lambda: download_sionna_scenes(folder=sionna_folder), number=1) < 1
    assert (
        timeit(lambda: download_sionna_scenes(folder=str(sionna_folder)), number=1) < 1
    )


@pytest.mark.slow
def test_download_sionna_scenes_existing_empty_folder(empty_folder: Path) -> None:
    download_sionna_scenes(folder=empty_folder, cached=False)


def test_download_sionna_scenes_existing_non_empty_folder(sionna_folder: Path) -> None:
    with pytest.raises(OSError, match=r"[Dd]irectory (is )?not empty"):
        download_sionna_scenes(folder=sionna_folder, cached=False)


def test_list_sionna_scenes(sionna_folder: Path) -> None:
    l_a = list_sionna_scenes(folder=sionna_folder)
    assert len(l_a) > 0
    l_b = list_sionna_scenes(folder=str(sionna_folder))

    for s_a, s_b in zip(l_a, l_b, strict=False):
        assert s_a == s_b


@pytest.mark.parametrize("scene_name", ["foo", "bar"])
def test_get_unexisting_sionna_scene(scene_name: str, sionna_folder: Path) -> None:
    with pytest.raises(ValueError, match="Cannot find scene_name"):
        _ = get_sionna_scene(scene_name, folder=sionna_folder)


@pytest.mark.parametrize("scene_name", ["box", "etoile", "munich"])
def test_get_existing_sionna_scene(scene_name: str, sionna_folder: Path) -> None:
    assert Path(get_sionna_scene(scene_name, folder=sionna_folder)).exists()
    assert Path(get_sionna_scene(scene_name, folder=str(sionna_folder))).exists()


class TestSionnaScene:
    def test_load_xml(self, sionna_folder: Path, subtests: SubTests) -> None:
        for scene_name in list_sionna_scenes(folder=sionna_folder):
            with subtests.test(scene_name=scene_name):
                file = get_sionna_scene(scene_name, folder=sionna_folder)
                scene = SionnaScene.load_xml(file)

                assert len(scene.shapes) > 0
                assert len(scene.materials) > 0

                for shape in scene.shapes.values():
                    assert shape.material_id in scene.materials
