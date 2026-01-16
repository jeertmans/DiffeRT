import chex
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import PRNGKeyArray

from differt.geometry import (
    rotation_matrix_along_x_axis,
    rotation_matrix_along_y_axis,
    rotation_matrix_along_z_axis,
)
from differt.scene import TriangleScene
from differt.utils import sample_points_in_bounding_box

from .generators import random_scene
from .submodels import Flow, ObjectsEncoder


class TestObjectsEncoder:
    @pytest.mark.parametrize("degenerate", [False, True])
    def test_invariance(
        self,
        degenerate: bool,
        objects_encoder: ObjectsEncoder,
        key: PRNGKeyArray,
    ) -> None:
        scene_key, transform_key = jr.split(key, 2)

        # 1 - Generate random scene
        scene = random_scene(key=scene_key)

        if degenerate:
            # Set RX-TX aligned with z-axis
            scene = eqx.tree_at(
                lambda s: s.receivers,
                scene,
                scene.transmitters.at[-1].set(scene.receivers[-1]),
            )

        # 2 - Compute embeddings
        e = objects_encoder(scene)

        # 3 - Apply random translation, scaling, and rotation to scene
        t_key, s_key, r_key = jr.split(transform_key, 3)

        # 3a - Test Translation Invariance
        t = sample_points_in_bounding_box(scene.mesh.bounding_box, key=t_key)
        scene_t = scene.translate(t)
        e_t = objects_encoder(scene_t)
        chex.assert_trees_all_close(
            e, e_t, atol=1e-5, custom_message="Translation invariance failed"
        )

        # 3b - Test Scaling Invariance
        s = jnp.exp(jr.normal(s_key))
        scene_s = scene.scale(s)
        e_s = objects_encoder(scene_s)
        chex.assert_trees_all_close(
            e, e_s, atol=1e-5, custom_message="Scaling invariance failed"
        )

        # 3c - Test Rotation Invariance (along z-axis)
        rot_z_angle = jr.uniform(r_key, minval=-jnp.pi, maxval=jnp.pi)
        R = rotation_matrix_along_z_axis(rot_z_angle)

        scene_r = scene.rotate(R)
        e_r = objects_encoder(scene_r)

        chex.assert_trees_all_close(
            e, e_r, atol=1e-5, custom_message="Rotation invariance failed"
        )

        # 3d - Test Combined Transformations
        scene_combined = scene.translate(t).scale(s).rotate(R)
        e_combined = objects_encoder(scene_combined)
        chex.assert_trees_all_close(
            e,
            e_combined,
            atol=1e-5,
            custom_message="Combined transformations invariance failed",
        )

        # 4 - Rotation invariance around x- and y-axes: expected failures
        rot_y_angle = jr.uniform(r_key, minval=-jnp.pi, maxval=jnp.pi)
        R = rotation_matrix_along_y_axis(rot_y_angle)

        scene_r = scene.rotate(R)
        e_r = objects_encoder(scene_r)

        with pytest.raises(AssertionError, match="Rotation invariance failed"):
            chex.assert_trees_all_close(
                e, e_r, atol=1e-5, custom_message="Rotation invariance failed"
            )

        rot_x_angle = jr.uniform(r_key, minval=-jnp.pi, maxval=jnp.pi)
        R = rotation_matrix_along_x_axis(rot_x_angle)

        scene_r = scene.rotate(R)
        e_r = objects_encoder(scene_r)

        with pytest.raises(AssertionError, match="Rotation invariance failed"):
            chex.assert_trees_all_close(
                e, e_r, atol=1e-5, custom_message="Rotation invariance failed"
            )


class TestFlow:
    def test_flow(
        self,
        order: int,
        inference: bool,
        flow: Flow,
        scene: TriangleScene,
        key: PRNGKeyArray,
    ) -> None:
        mask = scene.mesh.mask
        if mask is None:
            mask = jnp.array([], dtype=bool)
        inactive_objects = jnp.argwhere(mask == False)
        for sample_key in jr.split(key, 100):
            partial_path_candidate = -jnp.ones(order, dtype=int)
            last_object = jnp.array(-1)
            parent_flow_key, key = jr.split(sample_key)
            parent_flows = flow(
                scene,
                partial_path_candidate,
                last_object,
                inference=inference,
                key=parent_flow_key,
            )

            for i, key in enumerate(jr.split(key, order)):
                edge_flow_key, action_key = jr.split(key)

                action = jr.categorical(
                    action_key, logits=jnp.log(parent_flows)
                )
                partial_path_candidate = partial_path_candidate.at[i].set(
                    action
                )
                last_object = action

                edge_flows = flow(
                    scene,
                    partial_path_candidate,
                    last_object,
                    inference=inference,
                    key=edge_flow_key,
                )
                assert edge_flows[last_object] == 0, (
                    f"Flow should be zero for last object, got: {edge_flows[last_object]}"
                )
                assert (edge_flows >= 0).all(), (
                    f"Flow should be non-negative, got: {edge_flows}"
                )
                assert (edge_flows[inactive_objects] == 0).all(), (
                    f"Flow should be zero for inactive objects, got: {edge_flows[inactive_objects]}"
                )

                parent_flows = edge_flows
