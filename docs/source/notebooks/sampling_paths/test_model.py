import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

from differt.scene import TriangleScene

from .model import Model


class TestModel:
    def test_model(
        self, model: Model, scene: TriangleScene, key: PRNGKeyArray
    ) -> None:
        mask = scene.mesh.mask
        if mask is None:
            mask = jnp.array([], dtype=bool)
        inactive_objects = jnp.argwhere(~mask)
        for sample_key in jr.split(key, 100):
            path_candidate = model(scene, inference=True, key=sample_key)
            # model should never generate a path that contains the same object twice in a row
            assert not (path_candidate[:-1] == path_candidate[1:]).any(), (
                f"Path candidate should not contain the same object twice in a row, got: {path_candidate}"
            )
            # model should never generate a path that contains inactive objects
            assert not jnp.isin(inactive_objects, path_candidate).any(), (
                f"Path candidate should not contain inactive objects, got: {path_candidate}, but inactive objects are: {inactive_objects}"
            )
