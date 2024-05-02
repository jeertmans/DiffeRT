import pytest

from differt.scene.sionna import get_sionna_scene, list_sionna_scenes
from differt.scene.triangle_scene import TriangleScene
from differt_core.scene.sionna import SionnaScene


class TestTriangleScene:
    @pytest.mark.parametrize("scene_name", list_sionna_scenes())
    def test_load_xml(self, scene_name: str) -> None:
        if scene_name == "etoile":
            pytest.xfail("'etoile' scene currently fails to be loaded from XML")
        # Sionne scenes are all triangle scenes.
        file = get_sionna_scene(scene_name)
        scene = TriangleScene.load_xml(file)
        sionna_scene = SionnaScene.load_xml(file)

        assert len(scene.mesh_ids) == len(sionna_scene.shapes)
