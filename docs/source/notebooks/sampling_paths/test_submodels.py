import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

from differt.scene import TriangleScene

from .submodels import Flows, ObjectsEncoder, SceneEncoder, StateEncoder
from .utils import geometric_transformation, unpack_scene


class TestFlows:
    def test_flows(
        self,
        order: int,
        inference: bool,
        objects_encoder: ObjectsEncoder,
        scene_encoder: SceneEncoder,
        state_encoder: StateEncoder,
        flows: Flows,
        scene: TriangleScene,
        key: PRNGKeyArray,
    ) -> None:
        xyz = geometric_transformation(*unpack_scene(scene))
        objects_embeds = objects_encoder(xyz, active_objects=scene.mesh.mask)
        scene_embeds = scene_encoder(
            objects_embeds, active_objects=scene.mesh.mask
        )
        state_embeds = state_encoder(
            -jnp.ones(order, dtype=int),
            objects_embeds,
            active_objects=scene.mesh.mask,
        )
        mask = scene.mesh.mask
        if mask is None:
            mask = jnp.array([], dtype=bool)
        inactive_objects = jnp.argwhere(~mask)
        for sample_key in jr.split(key, 100):
            partial_path_candidate = -jnp.ones(order, dtype=int)
            last_object = jnp.array(-1)
            parent_flow_key, key = jr.split(sample_key)
            parent_flows = flows(
                objects_embeds,
                scene_embeds,
                state_embeds,
                last_object,
                active_objects=scene.mesh.mask,
                inference=inference,
                key=parent_flow_key,
            )

            for i, flow_key in enumerate(jr.split(key, order)):
                edge_flow_key, action_key = jr.split(flow_key)

                action = jr.categorical(
                    action_key, logits=jnp.log(parent_flows)
                )
                partial_path_candidate = partial_path_candidate.at[i].set(
                    action
                )
                last_object = action

                edge_flows = flows(
                    objects_embeds,
                    scene_embeds,
                    state_embeds,
                    last_object,
                    active_objects=scene.mesh.mask,
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
