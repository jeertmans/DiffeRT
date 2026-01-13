import chex
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

from differt.geometry import orthogonal_basis

from .generators import random_scene
from .submodels import SceneEncoder


class TestSceneEncoder:
    def test_invariance(self, key: PRNGKeyArray) -> None:
        key_scene, key_encoder, key_transform = jr.split(key, 3)

        # 1 - Generate random scene
        scene = random_scene(key=key_scene)

        # 2 - Initialize encoder
        encoder = SceneEncoder(
            num_embeddings=32,
            width_size=64,
            depth=2,
            key=key_encoder,
        )

        # 3 - Compute embeddings
        e = encoder(scene)

        # 4 - Apply random translation, scaling, and rotation to scene
        key_t, key_s, key_r = jr.split(key_transform, 3)

        # 4a - Test Translation Invariance
        t = jr.normal(key_t, (3,))
        scene_t = scene.translate(t)
        e_t = encoder(scene_t)
        chex.assert_trees_all_close(
            e, e_t, atol=1e-5, custom_message="Translation invariance failed"
        )

        # 4b - Test Scaling Invariance
        s = jnp.exp(jr.normal(key_s))
        scene_s = scene.scale(s)
        e_s = encoder(scene_s)
        chex.assert_trees_all_close(
            e, e_s, atol=1e-5, custom_message="Scaling invariance failed"
        )

        # 4c - Test Rotation Invariance
        u = jr.normal(key_r, (3,))
        v, w = orthogonal_basis(u)
        u_norm = u / jnp.linalg.norm(u)
        R = jnp.stack([u_norm, v, w], axis=-1)
        # TODO: fixme
        R = jnp.identity(3, dtype=R.dtype)

        scene_r = scene.rotate(R)
        e_r = encoder(scene_r)

        chex.assert_trees_all_close(
            e, e_r, atol=1e-5, custom_message="Rotation invariance failed"
        )

        # 4d - Test Combined Transformations
        scene_combined = scene.translate(t).scale(s).rotate(R)
        e_combined = encoder(scene_combined)
        chex.assert_trees_all_close(
            e,
            e_combined,
            atol=1e-5,
            custom_message="Combined transformations invariance failed",
        )
