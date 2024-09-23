import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped


@jaxtyped(typechecker=typechecker)
class PlanarMirrorsSetup(eqx.Module):
    from_vertices: Float[Array, "*batch 3"]
    to_vertices: Float[Array, "*batch 3"]
    mirror_vertices: Float[Array, "num_mirrors 3"]
    mirror_normals: Float[Array, "num_mirrors 3"]
    paths: Float[Array, "*batch num_mirrors 3"]

    def broadcast_to(self, *batch: int) -> "PlanarMirrorsSetup":
        return type(self)(
            from_vertices=jnp.broadcast_to(self.from_vertices, (*batch, 3)),
            to_vertices=jnp.broadcast_to(self.to_vertices, (*batch, 3)),
            mirror_vertices=self.mirror_vertices,
            mirror_normals=self.mirror_normals,
            paths=jnp.broadcast_to(self.paths, (*batch, *self.mirror_vertices.shape)),
        )

    def add_noeffect_noise(
        self, scale: float = 0.0, *, key: PRNGKeyArray
    ) -> "PlanarMirrorsSetup":
        num_mirrors = self.mirror_normals.shape[0]
        key_sign, key_shift = jax.random.split(key, 2)

        # Randomly shifting mirror origins has no effect as long as it is perpendicular to their normal direction
        shift = jax.random.normal(key_shift, shape=(num_mirrors, 3)) * scale
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
            key_sign, jnp.array([+1.0, -1.0]), shape=(num_mirrors,)
        )
        return eqx.tree_at(
            lambda setup: setup.mirror_normals,
            setup,
            self.mirror_normals * sign[:, None],
        )
