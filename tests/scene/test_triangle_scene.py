import pytest

from differt.scene.sionna import SionnaScene, get_sionna_scene, list_sionna_scenes
from differt.scene.triangle_scene import TriangleScene


class TestTriangleScene:
    @pytest.mark.parametrize("scene_name", list_sionna_scenes())
    def test_load_xml(self, scene_name: str) -> None:
        # Sionne scenes are all triangle scenes.
        file = get_sionna_scene(scene_name)
        scene = TriangleScene.load_xml(file)
        sionna_scene = SionnaScene.load_xml(file)

        assert len(scene.meshes) == len(sionna_scene.shapes)
