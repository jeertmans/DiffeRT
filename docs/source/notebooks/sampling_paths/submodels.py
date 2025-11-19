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

    linear: eqx.nn.Linear

    def __init__(
        self,
        order: int,
        num_embeddings: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.linear = eqx.nn.Linear(
            in_features=num_embeddings * order,
            out_features=num_embeddings,
            use_bias=False,
            key=key,
        )

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, path_candidate_object_embeds: Float[Array, "order num_embeddings"]
    ) -> Float[Array, "num_embeddings"]:
        return self.linear(path_candidate_object_embeds.reshape(-1))


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
            in_size=num_embeddings * (order + 1),
            out_size="scalar",
            width_size=num_embeddings,
            activation=jax.nn.leaky_relu,
            depth=depth,
            key=head_key,
        )
        self.dropout_rate = dropout_rate
        self.inference = inference
        self.log_probabilities = log_probabilities

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
        # exp(-100) ~= 0
        stopped_flow = -100.0 if self.log_probabilities else 0.0
        object_embeds = self.scene_encoder(scene)
        pc_object_embeds = jnp.take(
            object_embeds, partial_path_candidate, axis=0, fill_value=0
        )
        pc_embeds = pc_object_embeds.reshape(-1)
        # pc_embeds = self.state_encoder(pc_object_embeds)
        pc_embeds = jnp.tile(pc_embeds, (object_embeds.shape[0], 1))

        flows = jax.vmap(self.head)(
            jnp.concat((object_embeds, pc_embeds), axis=1)
        )

        if scene.mesh.mask is not None:
            flows = jnp.where(scene.mesh.mask, flows, stopped_flow)

        # Stop flow from flowing to the same object twice in a row
        i = jnp.argwhere(
            partial_path_candidate == -1, size=1, fill_value=-1
        ).reshape(())
        last_object = partial_path_candidate.at[i - 1].get(
            wrap_negative_indices=False, fill_value=-1
        )
        flows = flows.at[last_object].set(
            stopped_flow, wrap_negative_indices=False
        )

        if inference is None:
            inference = self.inference
        if inference:
            return flows
        if key is None:
            msg = "Argument 'key' cannot be 'None' when not running in inference mode."
            raise ValueError(msg)
        if not self.log_probabilities:
            flows = jnp.exp(flows)

        q = 1 - jax.lax.stop_gradient(self.dropout_rate)
        mask = jr.bernoulli(key, q, flows.shape)
        return jnp.where(mask, flows, stopped_flow)


class Z(eqx.Module):
    scene_encoder: SceneEncoder
    phi: eqx.nn.Linear
    alpha: eqx.nn.Linear
    rho: eqx.nn.Linear

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
        self.phi = eqx.nn.Linear(
            in_features=num_embeddings,
            out_features=num_embeddings,
            key=phi_key,
        )
        self.alpha = eqx.nn.Linear(
            in_features=num_embeddings,
            out_features="scalar",
            key=alpha_key,
        )
        self.rho = eqx.nn.Linear(
            in_features=num_embeddings,
            out_features="scalar",
            key=rho_key,
        )

    def __call__(self, scene: TriangleScene) -> Float[Array, " "]:
        x = self.scene_encoder(scene)
        x_phi = jax.vmap(self.phi)(x)
        x_alpha = jax.nn.softmax(jax.vmap(self.alpha)(x))
        out = self.rho((x_alpha[:, None] * x_phi).sum(axis=0))
        # Clip output to positive values only
        # because it makes no sense to predict a negative
        # number of valid paths
        return jax.nn.relu(out)
