import pytest

from differt.scene.sionna import (
    download_sionna_scenes,
    get_sionna_scene,
    list_sionna_scenes,
)
from differt_core.scene.sionna import SionnaScene

download_sionna_scenes()


@pytest.mark.parametrize("scene_name", ("foo", "bar"))
def test_get_unexisting_sionna_scene(scene_name: str) -> None:
    with pytest.raises(ValueError, match="Cannot find scene_name"):
        _ = get_sionna_scene(scene_name)


class TestSionnaScene:
    @pytest.mark.parametrize("scene_name", list_sionna_scenes())
    def test_load_xml(self, scene_name: str) -> None:
        if scene_name == "etoile":
            pytest.xfail("'etoile' scene currently fails to be loaded from XML")
        file = get_sionna_scene(scene_name)
        scene = SionnaScene.load_xml(file)

        assert len(scene.shapes) > 0
        assert len(scene.materials) > 0

        for shape in scene.shapes.values():
            assert shape.material_id in scene.materials
