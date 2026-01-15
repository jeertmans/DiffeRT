import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from beartype import beartype as typechecker
from jaxtyping import (
    Array,
    Float,
    Int,
    PRNGKeyArray,
    jaxtyped,
)

from differt.geometry import normalize, orthogonal_basis
from differt.rt import triangles_visible_from_vertices
from differt.scene import (
    TriangleScene,
)


def basis_for_canonical_frame(
    tx: Float[Array, "3"], rx: Float[Array, "3"]
) -> tuple[Float[Array, "3 3"], Float[Array, " "]]:
    """
    Compute the basis for the canonical frame where the z-axis is aligned with the vector from tx to rx.

    Args:
        tx: Transmitter position.
        rx: Receiver position.

    Returns:
        A tuple containing:
            - A 3x3 array representing the rotation matrix to the canonical frame.
            - A scalar representing the scale (distance between tx and rx).
    """
    u, scale = normalize(rx - tx)
    v, w = orthogonal_basis(u)
    return jnp.stack((v, w, u)), scale


class SceneEncoder(eqx.Module):
    """Generate embeddings from triangle vertices."""

    mlp: eqx.nn.MLP

    def __init__(
        self,
        num_embeddings: int,
        width_size: int,
        depth: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.mlp = eqx.nn.MLP(
            3 * 3,
            out_size=num_embeddings,
            width_size=width_size,
            depth=depth,
            key=key,
        )

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        scene: TriangleScene,
    ) -> Float[Array, "num_triangles num_embeddings"]:
        rx = scene.receivers.reshape(3)
        tx = scene.transmitters.reshape(3)
        triangle_vertices = scene.mesh.triangle_vertices
        basis, scale = basis_for_canonical_frame(tx, rx)
        triangle_vertices = (
            scene.mesh
            .translate(-tx)
            .scale(1 / scale)
            .rotate(basis)
            .triangle_vertices
        )
        xyz = triangle_vertices.reshape(-1, 3 * 3)
        return jax.vmap(self.mlp)(xyz)


class StateEncoder(eqx.Module):
    """Generate embeddings for a (partial) path candidate."""

    positional_encoding: Float[Array, "order"]

    def __init__(
        self,
        order: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.positional_encoding = jr.uniform(key, (order,))

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        partial_path_candidate: Int[Array, " order"],
        objects_embeds: Float[Array, "num_objects num_embeddings"],
    ) -> Float[Array, "num_embeddings"]:
        # [order num_embeddings]
        query = (
            jax.nn.one_hot(partial_path_candidate, objects_embeds.shape[0])
            @ objects_embeds
        )
        # [order]
        weights = (
            self.positional_encoding
        )  # , where=partial_path_candidate != -1)
        weights = jnp.broadcast_to(weights[:, None], query.shape)
        print(f"{query.shape=}")
        print(f"{weights.shape=}")
        return jnp.average(query, axis=0, weights=weights)


class Flow(eqx.Module):
    order: int = eqx.field(static=True)

    scene_encoder: SceneEncoder
    state_encoder: StateEncoder

    # TODO: add DeepSet to encode all objects into one features vector
    head: eqx.nn.MLP
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        *,
        order: int,
        num_embeddings: int,
        width_size: int,
        depth: int,
        dropout_rate: float = 0.05,
        inference: bool = False,
        key: PRNGKeyArray,
    ) -> None:
        scene_enc_key, state_enc_key, head_key = jr.split(key, 3)
        self.scene_encoder = SceneEncoder(
            num_embeddings=num_embeddings,
            width_size=width_size,
            depth=depth,
            key=scene_enc_key,
        )
        self.state_encoder = StateEncoder(
            order=order,
            key=state_enc_key,
        )
        self.head = eqx.nn.MLP(
            in_size=num_embeddings * 3 + 3,
            out_size="scalar",
            width_size=num_embeddings,
            activation=jax.nn.leaky_relu,
            depth=depth,
            key=head_key,
        )
        self.dropout = eqx.nn.Dropout(dropout_rate, inference=inference)

        self.order = order

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        scene: TriangleScene,
        partial_path_candidate: Int[Array, " order"],
        last_object: Int[Array, " "],
        *,
        inference: bool | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " num_objects"]:
        """
        Compute unnormalized probabilities (flows) for selecting the next object in the path
        given the current partial path candidate and the scene.

        Args:
            scene: The scene containing the objects and their geometry.
            partial_path_candidate: An array representing the current partial path candidate.
            last_object: Last object inserted in the partial path candidate.
            inference: Whether to run in inference mode (disables dropout).
            key: PRNG key for randomness (used in dropout), only required if not in inference mode.
        """
        # [num_objects num_embeddings]
        objects_embeds = self.scene_encoder(scene)
        # TODO: True DeepSets model to encode the entire scene
        scene_embeds = jnp.mean(
            objects_embeds,
            axis=0,
            where=scene.mesh.mask[:, None]
            if scene.mesh.mask is not None
            else None,
        )
        # [num_embeddings]
        pc_embeds = self.state_encoder(partial_path_candidate, objects_embeds)

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
            lambda object_embeds, pc_embeds, scene_embeds, relative_vec: (
                self.head(
                    jnp.concat(
                        (object_embeds, pc_embeds, scene_embeds, relative_vec),
                        axis=0,
                    )
                )
            ),
            in_axes=(0, None, None, 0),
        )(objects_embeds, pc_embeds, scene_embeds, relative_vectors)
        flows = jnp.exp(flows)

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
