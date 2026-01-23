import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray
from pytest_subtests import SubTests

from differt.scene import TriangleScene

from .submodels import Flows, ObjectsEncoder, SceneEncoder, StateEncoder
from .utils import geometric_transformation, random_scene, unpack_scene


class TestSceneEncoder:
    def test_scene_encoder(
        self,
        objects_encoder: ObjectsEncoder,
        scene_encoder: SceneEncoder,
        key: PRNGKeyArray,
        subtests: SubTests,
    ) -> None:
        for i, subtest_key in enumerate(jr.split(key, 10)):
            with subtests.test(i=i):
                scene_key, shuffles_key = jr.split(subtest_key)
                ref_scene = random_scene(key=scene_key)
                ref_xyz = geometric_transformation(*unpack_scene(ref_scene))
                ref_objects_embeds = objects_encoder(
                    ref_xyz, active_objects=ref_scene.mesh.mask
                )
                ref_scene_embeds = scene_encoder(
                    ref_objects_embeds, active_objects=ref_scene.mesh.mask
                )

                for shuffle_key in jr.split(shuffles_key, 3):
                    scene = eqx.tree_at(
                        lambda s: s.mesh,
                        ref_scene,
                        ref_scene.mesh.shuffle(key=shuffle_key),
                    )
                    xyz = geometric_transformation(*unpack_scene(scene))
                    objects_embeds = objects_encoder(
                        xyz, active_objects=scene.mesh.mask
                    )
                    scene_embeds = scene_encoder(
                        objects_embeds, active_objects=scene.mesh.mask
                    )

                    # Scene embeddings should be invariant to object order
                    chex.assert_trees_all_close(
                        scene_embeds,
                        ref_scene_embeds,
                        atol=1e-7,
                    )


class TestStateEncoder:
    def test_state_encoder(
        self,
        order: int,
        objects_encoder: ObjectsEncoder,
        state_encoder: StateEncoder,
        key: PRNGKeyArray,
        subtests: SubTests,
    ) -> None:
        for i, subtest_key in enumerate(jr.split(key, 10)):
            with subtests.test(i=i):
                scene_key, shuffles_key, path_candidate_key = jr.split(subtest_key, 3)
                ref_scene = random_scene(key=scene_key)
                ref_xyz = geometric_transformation(*unpack_scene(ref_scene))
                ref_objects_embeds = objects_encoder(
                    ref_xyz, active_objects=ref_scene.mesh.mask
                )
                num_objects = ref_xyz.shape[0]
                p = jax.nn.softmax(jnp.ones(num_objects), where=ref_scene.mesh.mask)
                path_candidate = jr.choice(
                    path_candidate_key, ref_xyz.shape[0], (order,), replace=False, p=p
                )

                for j in range(order):
                    ref_partial_path_candidate = path_candidate.at[j:].set(-1)

                    ref_state_embeds = state_encoder(
                        ref_partial_path_candidate,
                        objects_embeds=ref_objects_embeds,
                        active_objects=ref_scene.mesh.mask,
                    )

                    for shuffle_key in jr.split(shuffles_key, 3):
                        mesh, indices = ref_scene.mesh.shuffle(
                            return_indices=True, key=shuffle_key
                        )
                        rev_indices = jnp.empty_like(indices)
                        rev_indices = rev_indices.at[indices].set(
                            jnp.arange(num_objects)
                        )
                        partial_path_candidate = rev_indices.at[
                            ref_partial_path_candidate
                        ].get(wrap_negative_indices=False, mode="fill", fill_value=-1)
                        scene = eqx.tree_at(lambda s: s.mesh, ref_scene, mesh)
                        xyz = geometric_transformation(*unpack_scene(scene))
                        objects_embeds = objects_encoder(
                            xyz, active_objects=scene.mesh.mask
                        )
                        state_embeds = state_encoder(
                            partial_path_candidate,
                            objects_embeds=objects_embeds,
                            active_objects=scene.mesh.mask,
                        )
                        # State embeddings should be invariant to object order
                        chex.assert_trees_all_close(
                            state_embeds,
                            ref_state_embeds,
                            atol=1e-7,
                        )


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
        scene_embeds = scene_encoder(objects_embeds, active_objects=scene.mesh.mask)
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
            parent_flow_key, key = jr.split(sample_key)
            parent_flows = flows(
                objects_embeds,
                scene_embeds,
                state_embeds,
                active_objects=scene.mesh.mask,
                inference=inference,
                key=parent_flow_key,
            )

            for i, flow_key in enumerate(jr.split(key, order)):
                edge_flow_key, action_key = jr.split(flow_key)

                action = jr.categorical(action_key, logits=jnp.log(parent_flows))
                partial_path_candidate = partial_path_candidate.at[i].set(action)
                edge_flows = flows(
                    objects_embeds,
                    scene_embeds,
                    state_embeds,
                    active_objects=scene.mesh.mask,
                    inference=inference,
                    key=edge_flow_key,
                )
                edge_flows = edge_flows.at[action].set(0.0)
                assert edge_flows[action] == 0, (
                    f"Flow should be zero for last object, got: {edge_flows[action]}"
                )
                assert (edge_flows >= 0).all(), (
                    f"Flow should be non-negative, got: {edge_flows}"
                )
                assert (edge_flows[inactive_objects] == 0).all(), (
                    f"Flow should be zero for inactive objects, got: {edge_flows[inactive_objects]}"
                )

                parent_flows = edge_flows
