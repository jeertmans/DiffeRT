"""
Common antenna utilies.
"""
from abc import abstractmethod

from jaxtyping import jaxtyped, Float, Inexact, Array, ArrayLike
import jax.numpy as jnp
import equinox as eqx
from beartype import beartype as typechecker

from .constants import epsilon_0, c
from differt.geometry.utils import normalize

@jaxtyped(typechecker=typechecker)
class Antenna(eqx.Module):
    frequency: Float[ArrayLike, " "] = eqx.field(convert=jnp.asarray)
    center: Float[Array, "3"] = eqx.field(convert=jnp.asarray, default=jnp.array([0.0, 0.0, 0.0]))

    @abstractmethod
    def e_field(self, r: Float[Array, "*batch 3"], t: Float[Array, "*batch 3"] | None = None) -> Inexact[Array, "*batch 3"]:
        pass


    def plot_radiation_pattern(self):


@jaxtyped(typechecker=typechecker)
class Dipole(Antenna):
    height: Float[ArrayLike, " "] = eqx.field(convert=jnp.asarray, default=jnp.array(1.0))
    current: Float[ArrayLike, " "] = eqx.field(convert=jnp.asarray, default=jnp.array(1.0))
    orientation: Float[Array, "3"] = eqx.field(convert=jnp.asarray, default=jnp.array([0.0, 0.0, 1.0]))

    def e_field(self, r: Float[Array, "*batch 3"], t: Float[Array, " *batch"] | None = None) -> Inexact[Array, "*batch 3"]:
        dir, r = normalize(r - self.center, keepdims=True)
        w = self.frequency * 2 * jnp.pi
        k = w / c

        factor = - epsilon_0 * self.frequency * jnp.pi * self.current * self.height

        theta_dir = jnp.cross(dir, self.orientation)

        if t is not None:
            sin_pulsa = jnp.sin(w * t[..., None] - k * r)
        else:
            sin_pulsa = -jnp.sin(k * r)
        
        return factor * sin_pulsa * theta_dir

