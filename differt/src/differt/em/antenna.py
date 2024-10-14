"""
Common antenna utilies.
"""
from abc import abstractmethod
from typing import Any

from jaxtyping import jaxtyped, Float, Inexact, Array, ArrayLike
import jax.numpy as jnp
import equinox as eqx
from beartype import beartype as typechecker

from .constants import epsilon_0, c
from differt.geometry.utils import normalize, fibonacci_lattice
from differt.plotting import draw_surface, PlotOutput

@jaxtyped(typechecker=typechecker)
class Antenna(eqx.Module):
    frequency: Float[ArrayLike, " "] = eqx.field(converter=jnp.asarray)
    center: Float[Array, "3"] = eqx.field(converter=jnp.asarray, default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))

    @abstractmethod
    def e_field(self, r: Float[Array, "*batch 3"], t: Float[Array, "*batch 3"] | None = None) -> Inexact[Array, "*batch 3"]:
        pass


    def plot_radiation_pattern(self, num_points: int =int(1e2), **kwargs: Any) -> PlotOutput:
        """

        """
        u = jnp.linspace(0, 2 * jnp.pi, num_points)
        v = jnp.linspace(0, jnp.pi, num_points)
        x = jnp.outer(jnp.cos(u), jnp.sin(v))
        y = jnp.outer(jnp.sin(u), jnp.sin(v))
        z = jnp.outer(jnp.ones_like(u), jnp.cos(v))

        r = self.center + jnp.stack((x, y, z), axis=-1)

        e = self.e_field(r)

        gain_db = 10.0 * jnp.log10(jnp.sum(e * e, axis=-1, keepdims=True))
        gain_db = jnp.sum(e * e, axis=-1, keepdims=True)

        r *= self.center + (r - self.center) * gain_db

        return draw_surface(x=r[..., 0], y=r[..., 1], z=r[..., 2], colors=gain_db, **kwargs)


@jaxtyped(typechecker=typechecker)
class Dipole(Antenna):
    """
    A simple electrical dipole.

    Examples:
        The following example shows how to plot the radiation
        pattern (antenna gain in dB) at 1 meter.

        .. plotly::
            :fig-vars: fig

            >>> from differt.em.antenna import Dipole
            >>>
            >>> ant = Dipole(frequency=1e9)
            >>> fig = ant.plot_radiation_pattern(backend="plotly")
            >>> fig  # doctest: +SKIP
    """
    height: Float[ArrayLike, " "] = eqx.field(converter=jnp.asarray, default_factory=lambda: jnp.array(1.0))
    current: Float[ArrayLike, " "] = eqx.field(converter=jnp.asarray, default_factory=lambda: jnp.array(1.0))
    orientation: Float[Array, "3"] = eqx.field(converter=jnp.asarray, default_factory=lambda: jnp.array([0.0, 0.0, 1.0]))

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

