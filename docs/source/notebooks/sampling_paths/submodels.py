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

from differt.geometry import cartesian_to_spherical, normalize, orthogonal_basis
from differt.scene import (
    TriangleScene,
)


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
        rx = scene.transmitters.reshape(3)
        tx = scene.receivers.reshape(3)
        triangle_vertices = scene.mesh.triangle_vertices
        u, scale = normalize(rx - tx, keepdims=True)
        v, w = orthogonal_basis(u)
        basis = jnp.stack((v, w, u))  # tx->rx is the new 'z' direction
        triangle_vertices = (
            scene.mesh.translate(-tx)
            .scale(1 / scale)
            .rotate(basis)
            .triangle_vertices
        )
        xyz = triangle_vertices.reshape(-1, 9)
        return jax.vmap(self.mlp)(xyz)

        rpa = cartesian_to_spherical(xyz)
        # azimuth angle is equivariant with a rotation around the tx->rx axis
        # so we must transform it into a equivariant feature
        da = jnp.subtract.outer(rpa[:, 2], rpa[:, 2])
        two_pi = 2 * jnp.pi
        da = (da + two_pi) % two_pi
        sda = jnp.sum(da, axis=-1)
        rpa = rpa.at[:, 2].set(sda)
        return jax.vmap(self.mlp)(rpa.reshape(-1, 9))


class StateEncoder(eqx.Module):
    """Generate embeddings for a (partial) path candidates."""

    attention: eqx.nn.MultiheadAttention

    def __init__(
        self,
        order: int,
        num_embeddings: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=order,
            query_size=num_embeddings,
            key=key,
        )

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, partial_path_candidate: Int[Array, " order"], object_embeds: Float[Array, "num_objects num_embeddings"]
    ) -> Float[Array, "order num_embeddings"]:
        # [order num_embeddings]
        query = jax.nn.one_hot(partial_path_candidate, object_embeds.shape[0]) @ object_embeds
        # Self attention
        # TODO: should we use masking?
        # 1 - to ignore invalid objects
        # 2 - to ignore path candidate entries that are not yet filled
        # TODO: return embeddings from last candidate?
        return self.attention(query, query, query)


class Flow(eqx.Module):
    dropout_rate: float
    inference: bool
    log_probabilities: bool

    scene_encoder: SceneEncoder
    state_encoder: StateEncoder

    head: eqx.nn.MLP

    def __init__(
        self,
        *,
        order: int,
        num_embeddings: int,
        width_size: int,
        depth: int,
        dropout_rate: float = 0.05,
        inference: bool = False,
        log_probabilities: bool = True,
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
            num_embeddings=num_embeddings,
            key=state_enc_key,
        )
        self.head = eqx.nn.MLP(
            in_size=num_embeddings * 2,
            out_size="scalar",
            width_size=num_embeddings,
            activation=jax.nn.leaky_relu,
            depth=depth,
            key=head_key,
        )
        self.dropout_rate = dropout_rate
        self.inference = inference
        self.log_probabilities = log_probabilities
        if not log_probabilities:
            msg = "Non-log probabilities are not implemented yet."
            raise NotImplementedError(
                msg
            )

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        scene: TriangleScene,
        partial_path_candidate: Int[Array, " order"],
        *,
        inference: bool | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " num_objects"]:
        i = jnp.argwhere(
            partial_path_candidate == -1, size=1, fill_value=-1
        ).reshape(())
        last_object = partial_path_candidate.at[i - 1].get(
            wrap_negative_indices=False, fill_value=-1
        )
        # [num_objects num_embeddings]
        object_embeds = self.scene_encoder(scene)
        # [num_embeddings]
        pc_embeds = self.state_encoder(partial_path_candidate, object_embeds).at[i-1].get(wrap_negative_indices=False, fill_value=0.0)
        # [num_objects num_embeddings]
        pc_embeds = jnp.tile(pc_embeds, (object_embeds.shape[0], 1))
        # [num_objects]
        flows = jax.vmap(self.head)(
            jnp.concat((object_embeds, pc_embeds), axis=1)
        )
        # Stop flow from flowing to masked objects
        if scene.mesh.mask is not None:
            flows = jnp.where(scene.mesh.mask, flows, -100)

        # Stop flow from flowing to the same object twice in a row
        flows = flows.at[last_object].set(
            -100, wrap_negative_indices=False
        )

        # Convert to probabilities if needed
        if not self.log_probabilities:
            flows = jnp.exp(flows)

        if inference is None:
            inference = self.inference
        if inference:
            return flows
        if key is None:
            msg = "Argument 'key' cannot be 'None' when not running in inference mode."
            raise ValueError(msg)

        # Dropout
        q = 1.0 - jax.lax.stop_gradient(self.dropout_rate)
        mask = jr.bernoulli(key, q, flows.shape)
        return jnp.where(mask, flows, -100.0 if self.log_probabilities else 0.0)


class Z(eqx.Module):
    """Estimate the number of valid paths in a scene using a Deep Sets architecture."""
    scene_encoder: SceneEncoder
    """Transform-invariant encoder for scene objects."""
    phi: eqx.nn.MLP
    """Layer to transform object embeddings."""
    alpha: eqx.nn.MLP
    """Layer to compute attention weights over objects."""
    rho: eqx.nn.MLP
    """Layer to compute final output."""

    def __init__(
        self,
        num_embeddings: int,
        width_size: int,
        depth: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        scene_enc_key, phi_key, alpha_key, rho_key = jr.split(key, 4)
        self.scene_encoder = SceneEncoder(
            num_embeddings=num_embeddings,
            width_size=width_size,
            depth=depth,
            key=scene_enc_key,
        )
        self.phi = eqx.nn.MLP(
            in_size=num_embeddings,
            out_size=num_embeddings,
            width_size=width_size,
            depth=depth,
            key=phi_key,
        )
        self.alpha = eqx.nn.MLP(
            in_size=num_embeddings,
            out_size="scalar",
            width_size=width_size,
            depth=depth,
            key=alpha_key,
        )
        self.rho = eqx.nn.MLP(
            in_size=num_embeddings,
            out_size="scalar",
            width_size=width_size,
            depth=depth,
            key=rho_key,
        )

    @jaxtyped(typechecker=typechecker)
    def __call__(self, scene: TriangleScene) -> Float[Array, " "]:
        x = self.scene_encoder(scene)
        x_phi = jax.vmap(self.phi)(x)
        # Compute attention weights over objects, while ignoring masked objects
        x_alpha = jax.nn.softmax(jax.vmap(self.alpha)(x), where=scene.mesh.mask)
        out = self.rho(jnp.dot(x_alpha, x_phi))
        # Clip output to positive values only
        # because it makes no sense to predict a negative
        # number of valid paths
        return jnp.exp(out)
