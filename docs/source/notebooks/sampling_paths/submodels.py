import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from beartype import beartype as typechecker
from jaxtyping import (
    Array,
    Bool,
    Float,
    Int,
    PRNGKeyArray,
    jaxtyped,
)

from differt.geometry import normalize
from differt.rt import triangles_visible_from_vertices
from differt.scene import (
    TriangleScene,
)


def basis_for_canonical_frame(
    tx: Float[Array, "3"],
    rx: Float[Array, "3"],
) -> tuple[Float[Array, "3 3"], Float[Array, " "]]:
    """
    Compute the basis for the canonical frame where the z-axis is aligned with the projection of the tx-rx direction on the xy-plane.

    Args:
        tx: Transmitter position.
        rx: Receiver position.

    Returns:
        A tuple containing:
            - A 3x3 array representing the rotation matrix to the canonical frame.
            - A scalar representing the scale (distance between tx and rx).
    """
    w, scale = normalize(rx - tx)
    ref_axis = jnp.array([0.0, 0.0, 1.0])
    u, _ = normalize(jnp.cross(w, ref_axis))
    v, _ = normalize(jnp.cross(w, u))
    return jnp.stack((u, v, w)), scale


class ObjectsEncoder(eqx.Module):
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


class SceneEncoder(eqx.Module):
    """Generate scene embeddings from objects embeddings."""

    attention: eqx.nn.Linear
    rho: eqx.nn.MLP

    def __init__(
        self,
        num_embeddings: int,
        width_size: int,
        depth: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        att_key, rho_key = jr.split(key)
        self.attention = eqx.nn.Linear(
            num_embeddings,
            "scalar",
            key=att_key,
        )
        self.rho = eqx.nn.MLP(
            in_size=num_embeddings,
            out_size=num_embeddings,
            width_size=width_size,
            depth=depth,
            key=rho_key,
        )

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        objects_embeds: Float[Array, "num_objects num_embeddings"],
        active_objects: Bool[Array, "num_objects"] | None = None,
    ) -> Float[Array, " num_embeddings"]:
        # [num_objects]
        weights = jax.nn.softmax(
            jax.vmap(self.attention)(objects_embeds), where=active_objects
        )
        # [num_embeddings]
        weigthed_sum = jnp.dot(weights, objects_embeds)
        return self.rho(weigthed_sum)


class StateEncoder(eqx.Module):
    """Generate embeddings for a (partial) path candidate."""

    linear: eqx.nn.Linear
    cell: eqx.nn.GRUCell

    def __init__(
        self,
        num_embeddings: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        linear_key, cell_key = jr.split(key)
        self.linear = eqx.nn.Linear(
            num_embeddings,
            num_embeddings,
            key=linear_key,
        )
        self.cell = eqx.nn.GRUCell(
            num_embeddings,
            num_embeddings,
            key=cell_key,
        )

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        partial_path_candidate: Int[Array, " order"],
        objects_embeds: Float[Array, "num_objects num_embeddings"],
    ) -> Float[Array, " num_embeddings"]:
        def scan_fn(
            state: Float[Array, " num_embeddings"],
            object_idx: Int[Array, ""],
        ) -> tuple[Float[Array, " num_embeddings"], None]:
            new_state = self.cell(objects_embeds[object_idx], state)
            return jnp.where(object_idx != -1, new_state, state), None

        init_state = jnp.zeros((self.cell.hidden_size,))
        final_state, _ = jax.lax.scan(
            scan_fn,
            init_state,
            partial_path_candidate,
        )
        return self.linear(final_state)


class Flow(eqx.Module):
    order: int = eqx.field(static=True)

    objects_encoder: ObjectsEncoder
    scene_encoder: SceneEncoder
    state_encoder: StateEncoder

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
        objects_enc_key, scene_enc_key, state_enc_key, head_key = jr.split(
            key, 4
        )
        self.objects_encoder = ObjectsEncoder(
            num_embeddings=num_embeddings,
            width_size=width_size,
            depth=depth,
            key=objects_enc_key,
        )
        self.scene_encoder = SceneEncoder(
            num_embeddings=num_embeddings,
            width_size=width_size,
            depth=depth,
            key=scene_enc_key,
        )
        self.state_encoder = StateEncoder(
            num_embeddings=num_embeddings,
            key=state_enc_key,
        )
        self.head = eqx.nn.MLP(
            in_size=num_embeddings * 3,
            out_size="scalar",
            width_size=num_embeddings,
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
        objects_embeds = self.objects_encoder(scene)
        # [num_embeddings]
        scene_embeds = self.scene_encoder(
            objects_embeds, active_objects=scene.mesh.mask
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
            lambda object_embeds, pc_embeds, scene_embeds: self.head(
                jnp.concat(
                    (object_embeds, pc_embeds, scene_embeds),
                    axis=0,
                )
            ),
            in_axes=(0, None, None),
        )(objects_embeds, pc_embeds, scene_embeds)
        # Positive flows
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
