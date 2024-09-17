from pathlib import Path

import equinox as eqx
import jax.numpy as jnp

from differt.scene.sionna import (
    get_sionna_scene,
    list_sionna_scenes,
)
from differt.scene.triangle_scene import TriangleScene
from differt_core.scene.sionna import SionnaScene


class TestTriangleScene:
    def test_load_xml(self, sionna_folder: Path) -> None:
        # Sionne scenes are all triangle scenes.
        for scene_name in list_sionna_scenes(folder=sionna_folder):
            file = get_sionna_scene(scene_name, folder=sionna_folder)
            scene = TriangleScene.load_xml(file)
            sionna_scene = SionnaScene.load_xml(file)

            assert scene.mesh.object_bounds is not None
            assert len(scene.mesh.object_bounds) == len(sionna_scene.shapes)

    def test_plot(
        self,
        sionna_folder: Path,
    ) -> None:
        file = get_sionna_scene("simple_street_canyon", folder=sionna_folder)
        scene = TriangleScene.load_xml(file)

        tx = jnp.array([[0.0, 0.0, 0.0]])
        rx = jnp.array([[1.0, 1.0, 1.0]])

        scene = eqx.tree_at(lambda s: s.transmitters, scene, tx)
        scene = eqx.tree_at(lambda s: s.receivers, scene, rx)

        scene.plot()
