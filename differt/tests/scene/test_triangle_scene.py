import equinox as eqx
import jax.numpy as jnp
import pytest

from differt.scene.sionna import (
    download_sionna_scenes,
    get_sionna_scene,
    list_sionna_scenes,
)
from differt.scene.triangle_scene import TriangleScene
from differt_core.scene.sionna import SionnaScene

download_sionna_scenes()


class TestTriangleScene:
    @pytest.mark.parametrize("scene_name", list_sionna_scenes())
    def test_load_xml(self, scene_name: str) -> None:
        # Sionne scenes are all triangle scenes.
        file = get_sionna_scene(scene_name)
        scene = TriangleScene.load_xml(file)
        sionna_scene = SionnaScene.load_xml(file)

        assert len(scene.meshes) == len(sionna_scene.shapes)

    def test_plot(self) -> None:
        file = get_sionna_scene("simple_street_canyon")
        scene = TriangleScene.load_xml(file)

        tx = jnp.array([[0.0, 0.0, 0.0]])
        rx = jnp.array([[1.0, 1.0, 1.0]])

        scene = eqx.tree_at(lambda s: s.transmitters, scene, tx)
        scene = eqx.tree_at(lambda s: s.receivers, scene, rx)

        scene.plot()
