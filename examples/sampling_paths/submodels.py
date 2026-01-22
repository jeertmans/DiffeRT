import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Bool,
    Float,
    Int,
    PRNGKeyArray,
)


class ObjectsEncoder(eqx.Module):
    """Generate embeddings from triangle vertices."""

    out_size: int = eqx.field(static=True)

    mlp: eqx.nn.MLP

    def __init__(
        self,
        num_embeddings: int,
        width_size: int,
        depth: int,
        num_vertices_per_object: int = 3,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.out_size = num_embeddings

        self.mlp = eqx.nn.MLP(
            in_size=num_vertices_per_object * 3,
            out_size=num_embeddings,
            width_size=width_size,
            depth=depth,
            key=key,
        )

    def __call__(
        self,
        xyz: Float[Array, "num_objects num_vertices_per_object 3"],
        *,
        active_objects: Bool[Array, " num_objects"] | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "num_objects num_embeddings"]:
        del key
        embeds = jax.vmap(self.mlp)(xyz.reshape(xyz.shape[0], -1))
        if active_objects is not None:
            return jnp.where(active_objects[:, None], embeds, 0.0)
        return embeds


class SceneEncoder(eqx.Module):
    """Generate scene embeddings from objects embeddings."""

    out_size: int = eqx.field(static=True)

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
        self.out_size = num_embeddings
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

    def __call__(
        self,
        objects_embeds: Float[Array, "num_objects num_embeddings"],
        *,
        active_objects: Bool[Array, " num_objects"] | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " num_embeddings"]:
        del key
        # [num_objects]
        # weights = jax.nn.softmax(
        #    jax.vmap(self.attention)(objects_embeds), where=active_objects
        # )
        # [num_embeddings]
        # weigthed_sum = jnp.dot(weights, objects_embeds)
        return self.rho(
            objects_embeds.mean(
                axis=0,
                where=active_objects[:, None] if active_objects is not None else None,
            )
        )


class StateEncoder(eqx.Module):
    """Generate embeddings for a (partial) path candidate."""

    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        order: int,
        num_embeddings: int,
        *,
        key: PRNGKeyArray | None = None,
    ) -> None:
        del key
        self.out_size = order * num_embeddings

    def __call__(
        self,
        partial_path_candidate: Int[Array, " order"],
        objects_embeds: Float[Array, "num_objects num_embeddings"],
        *,
        active_objects: Bool[Array, "num_objects"] | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " out_size"]:
        del active_objects, key
        return (
            objects_embeds
            .at[partial_path_candidate]
            .get(mode="fill", wrap_negative_indices=False, fill_value=0.0)
            .reshape(self.out_size)
        )


class Flows(eqx.Module):
    """Compute unnormalized probabilities (flows) for each object."""

    mlp: eqx.nn.MLP
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        *,
        in_size: int,
        width_size: int,
        depth: int,
        dropout_rate: float,
        inference: bool = False,
        key: PRNGKeyArray,
    ) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size="scalar",
            width_size=width_size,
            depth=depth,
            activation=jax.nn.leaky_relu,
            final_activation=jnp.exp,
            key=key,
        )
        self.dropout = eqx.nn.Dropout(dropout_rate, inference=inference)

    def __call__(
        self,
        objects_embeds: Float[Array, "num_objects num_embeddings"],
        scene_embeds: Float[Array, " num_embeddings"],
        state_embeds: Float[Array, "  num_embeddings"],
        *,
        active_objects: Bool[Array, " num_objects"] | None = None,
        inference: bool | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " num_objects"]:
        """
        Compute unnormalized probabilities (flows) for selecting the next object in the path
        given the current partial path candidate and the scene.

        Args:
            objects_embeds: Embeddings for all objects in the scene.
            scene_embeds: Embeddings for the scene.
            state_embeds: Embeddings for the current partial path candidate.
            active_objects: Boolean array indicating which objects are active.
            inference: Whether to run in inference mode (disables dropout).
            key: PRNG key for randomness (used in dropout), only required if not in inference mode.

        Returns:
            Unnormalized probabilities (flows) for each object.
        """
        flows = jax.vmap(
            lambda object_embeds, scene_embeds, state_embeds: self.mlp(
                jnp.concat((object_embeds, scene_embeds, state_embeds))
            ),
            in_axes=(0, None, None),
        )(
            objects_embeds,
            scene_embeds,
            state_embeds,
        )
        flows = self.dropout(flows, inference=inference, key=key)
        if active_objects is not None:
            flows = jnp.where(active_objects, flows, 0.0)
        return flows
