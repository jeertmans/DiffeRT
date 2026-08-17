import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class PlanarMirrorsSetup(eqx.Module):
    from_vertices: Float[Array, "*#batch 3"]
    to_vertices: Float[Array, "*#batch 3"]
    mirror_vertices: Float[Array, "*#batch num_mirrors 3"]
    mirror_normals: Float[Array, "*#batch num_mirrors 3"]
    paths: Float[Array, "*#batch num_mirrors 3"]

    def broadcast_to(self, *batch: int) -> "PlanarMirrorsSetup":
        num_mirrors = self.mirror_vertices.shape[-2]
        return type(self)(
            from_vertices=jnp.broadcast_to(self.from_vertices, (*batch, 3)),
            to_vertices=jnp.broadcast_to(self.to_vertices, (*batch, 3)),
            mirror_vertices=jnp.broadcast_to(
                self.mirror_vertices, (*batch, num_mirrors, 3)
            ),
            mirror_normals=jnp.broadcast_to(
                self.mirror_normals, (*batch, num_mirrors, 3)
            ),
            paths=jnp.broadcast_to(self.paths, (*batch, num_mirrors, 3)),
        )

    def add_noeffect_noise(
        self, scale: float = 0.0, *, key: PRNGKeyArray
    ) -> "PlanarMirrorsSetup":
        key_sign, key_shift = jax.random.split(key, 2)

        # Randomly shifting mirror origins has no effect as long as it is perpendicular to their normal direction
        shift = jax.random.normal(key_shift, shape=self.mirror_vertices.shape) * scale
        shift = (
            shift
            - jnp.sum(shift * self.mirror_normals, axis=-1, keepdims=True)
            * self.mirror_normals
        )
        setup = eqx.tree_at(
            lambda setup: setup.mirror_vertices,
            self,
            self.mirror_vertices + shift,
        )

        # Randomly flipping normals has no effect
        sign = jax.random.choice(
            key_sign, jnp.array([+1.0, -1.0]), shape=self.mirror_vertices.shape[:-1]
        )
        return eqx.tree_at(
            lambda setup: setup.mirror_normals,
            setup,
            self.mirror_normals * sign[..., None],
        )
