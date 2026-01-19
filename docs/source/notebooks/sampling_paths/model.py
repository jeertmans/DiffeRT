from typing import Literal, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Float,
    Int,
    PRNGKeyArray,
)

from differt.scene import (
    TriangleScene,
)

from .submodels import Flows, ObjectsEncoder, SceneEncoder, StateEncoder
from .utils import geometric_transformation, unpack_scene


class Model(eqx.Module):
    order: int = eqx.field(static=True)

    objects_encoder: ObjectsEncoder
    scene_encoder: SceneEncoder
    state_encoder: StateEncoder
    flows: Flows

    inference: bool

    def __init__(
        self,
        *,
        order: int,
        num_embeddings: int,
        width_size: int,
        depth: int,
        num_vertices_per_object: int = 3,
        dropout_rate: float = 0.05,
        inference: bool = False,
        key: PRNGKeyArray,
    ) -> None:
        self.order = order
        self.inference = inference

        self.objects_encoder = ObjectsEncoder(
            num_embeddings=num_embeddings,
            width_size=width_size,
            depth=depth,
            num_vertices_per_object=num_vertices_per_object,
            key=key,
        )
        self.scene_encoder = SceneEncoder(
            num_embeddings=num_embeddings,
            width_size=width_size,
            depth=depth,
            key=key,
        )
        self.state_encoder = StateEncoder(
            order=order,
            num_embeddings=num_embeddings,
            out_size=num_embeddings,
            key=key,
        )
        self.flows = Flows(
            in_size=num_embeddings * 3,
            width_size=width_size,
            depth=depth,
            dropout_rate=dropout_rate,
            inference=inference,
            key=key,
        )

    @overload
    def __call__(
        self,
        scene: TriangleScene,
        *,
        inference: Literal[True],
        key: PRNGKeyArray,
    ) -> Int[Array, " order"]: ...

    @overload
    def __call__(
        self,
        scene: TriangleScene,
        *,
        inference: Literal[False],
        key: PRNGKeyArray,
    ) -> tuple[Int[Array, " order"], Float[Array, "num_objects order"]]: ...

    def __call__(
        self,
        scene: TriangleScene,
        *,
        inference: bool | None = None,
        key: PRNGKeyArray,
    ) -> (
        Int[Array, " order"]
        | tuple[Int[Array, " order"], Float[Array, "num_objects order"]]
    ):
        # [num_objects 3 3]
        xyz = geometric_transformation(*unpack_scene(scene))
        # [num_objects num_embeddings]
        objects_embeds = self.objects_encoder(
            xyz, active_objects=scene.mesh.mask
        )
        # [num_embeddings]
        scene_embeds = self.scene_encoder(
            objects_embeds, active_objects=scene.mesh.mask
        )
        # [order]
        partial_path_candidate = -jnp.ones(self.order, dtype=int)
        # [num_embeddings]
        state_embeds = self.state_encoder(
            partial_path_candidate,
            objects_embeds,
            active_objects=scene.mesh.mask,
        )

        last_object = jnp.array(-1)
        parent_flow_key, key = jr.split(key)
        parent_flows = self.flows(
            objects_embeds,
            scene_embeds,
            state_embeds,
            last_object,
            active_objects=scene.mesh.mask,
            inference=inference,
            key=parent_flow_key,
        )

        flows = []

        # For-loop is fine here because it will be unrolled at compile time
        for i, key in enumerate(jr.split(key, self.order)):
            flows.append(parent_flows)
            edge_flow_key, action_key = jr.split(key)

            last_object = jr.categorical(
                action_key, logits=jnp.log(parent_flows)
            )
            partial_path_candidate = partial_path_candidate.at[i].set(
                last_object
            )
            state_embeds = self.state_encoder(
                partial_path_candidate, objects_embeds
            )

            edge_flows = self.flows(
                objects_embeds,
                scene_embeds,
                state_embeds,
                last_object,
                active_objects=scene.mesh.mask,
                inference=inference,
                key=edge_flow_key,
            )

            parent_flows = edge_flows

        if inference or self.inference:
            return partial_path_candidate

        return partial_path_candidate, jnp.stack(flows, axis=-1)

        # [num_embeddings]
        state_embeds = self.state_encoder(
            partial_path_candidate, objects_embeds
        )

        # Calculate relative vectors from last object center
        rx = scene.receivers.reshape(3)
        tx = scene.transmitters.reshape(3)
        basis, scale = basis_for_canonical_frame(tx, rx)

        object_centers = scene.mesh.triangle_vertices.mean(axis=-2)
        # Transform centers to canonical frame
        object_centers = (object_centers - tx) / scale @ basis.T

        last_object_center = jnp.where(
            last_object != -1,
            object_centers[last_object],
            jnp.zeros(3),  # Since we translated by tx, tx is at origin (0,0,0)
        )

        # [num_objects 3]
        relative_vectors = object_centers - last_object_center

        # [num_objects]
        flows = jax.vmap(
            lambda object_embeds, pc_embeds, scene_embeds: self.head(
                jnp.concat(
                    (object_embeds, pc_embeds, scene_embeds),
                    axis=0,
                )
            ),
            in_axes=(0, None, None),
        )(objects_embeds, pc_embeds, scene_embeds)

        # Stop flow from flowing to masked objects
        mask = (
            jnp.ones_like(flows).astype(bool)
            if scene.mesh.mask is None
            else scene.mesh.mask
        )

        # Stop flow from flowing to same object again
        mask = mask.at[last_object].set(False, wrap_negative_indices=False)

        # Stop flow from flowing to unreachable objects
        object_centers = scene.mesh.triangle_vertices.mean(axis=-2)
        object_normals = scene.mesh.normals

        mask &= jnp.where(
            last_object == -1,
            triangles_visible_from_vertices(
                scene.transmitters, scene.mesh.triangle_vertices
            ),
            True,
        )

        flows = jnp.where(mask, flows, 0.0)

        if self.order == 1 and False:
            tx_to_object = object_centers - scene.transmitters.reshape(3)
            rx_to_object = object_centers - scene.receivers.reshape(3)

            same_side_of_objects = jnp.sign(
                jnp.sum(tx_to_object * object_normals, axis=-1)
            ) == jnp.sign(jnp.sum(rx_to_object * object_normals, axis=-1))

            flows = jnp.where(same_side_of_objects, flows, 0.0)
            r = jnp.linalg.norm(tx_to_object, axis=-1) + jnp.linalg.norm(
                rx_to_object, axis=-1
            )
            flows *= jax.nn.softmax(-r, where=scene.mesh.mask)

        return self.dropout(flows, key=key, inference=inference)
