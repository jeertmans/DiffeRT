from typing import Literal, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    ArrayLike,
    Float,
    Int,
    PRNGKeyArray,
)

from differt.scene import (
    TriangleScene,
)

from .metrics import reward_fn
from .submodels import Flows, ObjectsEncoder, SceneEncoder, StateEncoder
from .utils import geometric_transformation, unpack_scene


def scan(f, init, xs):
    carry = init
    ys = []
    for x in zip(*xs):
        carry, y = f(carry, x)
        if y is not None:
            ys.append(y)
    return carry, jnp.stack(ys) if ys else None


class Model(eqx.Module):
    order: int = eqx.field(static=True)

    objects_encoder: ObjectsEncoder
    scene_encoder: SceneEncoder
    state_encoder: StateEncoder
    flows: Flows

    epsilon: Float[Array, ""]

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
        epsilon: Float[ArrayLike, ""] = 0.5,
        inference: bool = False,
        key: PRNGKeyArray,
    ) -> None:
        self.order = order

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
            key=key,
        )
        self.flows = Flows(
            in_size=self.objects_encoder.out_size
            + self.scene_encoder.out_size
            + self.state_encoder.out_size,
            width_size=width_size,
            depth=depth,
            dropout_rate=dropout_rate,
            inference=inference,
            key=key,
        )

        self.epsilon = jnp.asarray(epsilon)

        self.inference = inference

    @overload
    def __call__(
        self,
        scene: TriangleScene,
        *,
        replay: Int[Array, " order"] | None = None,
        inference: Literal[True],
        key: PRNGKeyArray,
    ) -> Int[Array, " order"]: ...

    @overload
    def __call__(
        self,
        scene: TriangleScene,
        *,
        replay: Int[Array, " order"] | None = None,
        inference: Literal[False],
        key: PRNGKeyArray,
    ) -> tuple[Int[Array, " order"], Float[Array, ""], Float[Array, ""]]: ...

    def __call__(
        self,
        scene: TriangleScene,
        *,
        replay: Int[Array, " order"] | None = None,
        inference: bool | None = None,
        key: PRNGKeyArray,
    ) -> (
        Int[Array, " order"]
        | tuple[Int[Array, " order"], Float[Array, ""], Float[Array, ""]]
    ):
        """
        Sample a path candidate in the given scene.

        Args:
            scene: The scene in which to sample the path candidate.
            replay: If provided, replay this path candidate instead of sampling it randomly.
            inference: Whether to run in inference mode (disables epsilon-greedy uniform sampling and dropout).
            key: PRNG key for randomness.

        Returns:
            If inference is True, returns the sampled path candidate as an array of integers.
            If inference is False, returns a tuple containing the sampled path candidate, the loss value, and the reward.
        """
        inference = self.inference if inference is None else inference

        # [num_objects 3 3]
        xyz = geometric_transformation(*unpack_scene(scene))
        num_objects = xyz.shape[0]
        # [num_objects num_embeddings]
        objects_embeds = self.objects_encoder(xyz, active_objects=scene.mesh.mask)
        # [num_embeddings]
        scene_embeds = self.scene_encoder(
            objects_embeds, active_objects=scene.mesh.mask
        )

        def scan_fn(
            carry: tuple[
                Int[Array, "order"],
                Float[Array, ""],
                Float[Array, "num_objects"],
                Int[Array, ""],
            ],
            x: tuple[Int[Array, ""], PRNGKeyArray],
        ) -> tuple[
            tuple[
                Int[Array, "order"],
                Float[Array, ""],
                Float[Array, "num_objects"],
                Int[Array, ""],
            ],
            Float[Array, ""],
        ]:

            partial_path_candidate, loss_value, edge_flows, previous_object = carry
            i, key = x

            next_object_key, next_edge_flows_key = jr.split(key)

            # Sample next object
            flow_policy = edge_flows
            if (
                not inference
            ):  # During training, we use uniform policy with probability epsilon
                epsilon_greedy_key, next_object_key = jr.split(next_object_key)
                uniform_policy = (
                    jnp.where(scene.mesh.mask, 1.0, 0.0)
                    if scene.mesh.mask is not None
                    else jnp.ones_like(flow_policy)
                )
                uniform_policy = uniform_policy.at[previous_object].set(
                    0.0, wrap_negative_indices=False
                )
                choose_uniform_policy = jr.bernoulli(
                    epsilon_greedy_key, self.epsilon
                ) | (edge_flows.sum() == 0.0)
                policy = jnp.where(
                    choose_uniform_policy,
                    uniform_policy,
                    flow_policy,
                )
            else:
                policy = flow_policy

            if replay is not None:
                next_object = replay[i]
            else:
                next_object = jr.choice(next_object_key, num_objects, p=policy)

            # Update state variables
            partial_path_candidate = partial_path_candidate.at[i].set(next_object)
            state_embeds = self.state_encoder(
                partial_path_candidate,
                objects_embeds,
                active_objects=scene.mesh.mask,
            )

            # Compute loss (flow mismatch)
            parent_flow = edge_flows[next_object]

            reward = jnp.where(
                i == self.order - 1,
                reward_fn(partial_path_candidate, scene),
                0.0,
            )

            edge_flows = jnp.where(
                i == self.order - 1,
                jnp.zeros_like(edge_flows),
                self
                .flows(
                    objects_embeds,
                    scene_embeds,
                    state_embeds,
                    active_objects=scene.mesh.mask,
                    inference=inference,
                    key=next_edge_flows_key,
                )
                .at[next_object]
                .set(0.0),
            )

            loss_value += (parent_flow - edge_flows.sum() - reward) ** 2

            return (
                partial_path_candidate,
                loss_value,
                edge_flows,
                next_object,
            ), reward

        init_edge_flows_key, scan_key = jr.split(key)

        init_path_candidate = -jnp.ones(self.order, dtype=int)
        init_loss_value = jnp.array(0.0)
        init_state_embeds = jnp.zeros(self.state_encoder.out_size)
        init_edge_flows = self.flows(
            objects_embeds,
            scene_embeds,
            init_state_embeds,
            active_objects=scene.mesh.mask,
            inference=inference,
            key=init_edge_flows_key,
        )

        (path_candidate, loss_value, _, _), rewards = scan(
            scan_fn,
            (
                init_path_candidate,
                init_loss_value,
                init_edge_flows,
                jnp.array(-1),
            ),
            (jnp.arange(self.order), jr.split(scan_key, self.order)),
        )

        if inference:
            return path_candidate

        return path_candidate, loss_value, rewards[-1]

        raise ValueError("Should never reach here")

        # [num_embeddings]
        state_embeds = self.state_encoder(partial_path_candidate, objects_embeds)

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
