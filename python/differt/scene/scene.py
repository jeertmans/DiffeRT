"""TODO."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from typeguard import typechecked as typechecker

from ..geometry.triange_mesh import TriangleMesh


@jaxtyped(typechecker=typechecker)
class Scene(eqx.Module):
    """
    A simple scene made of a mesh and transmitter / receiver
    coordinates.

    Args:
        tx: The transmitter coordinates.
        rx: The receiver coordinates.
    """

    vertices: Float[Array, "*tx_batch 3"] = eqx.field(converter=jnp.asarray)
    """The array of transmitter coordinates."""
    triangles: Float[Array, "*rx_batch 3"] = eqx.field(converter=jnp.asarray)
    """The array of receiver coordinates."""


@jaxtyped(typechecker=typechecker)
class TriangleScene(Scene, eqx.Module):
    """
    A simple scene made of triangles.

    Args:
        mesh: The triangle mesh.
    """

    mesh: TriangleMesh

    def compute_paths(
        self, from_vertices, to_vertices, min_order: int = 0, max_order: int = 1
    ) -> None:
        pass
