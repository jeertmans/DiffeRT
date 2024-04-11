"""Mesh geometry made of triangles and utilities."""

from collections.abc import Mapping
from typing import Any, Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

from .. import _core
from ..geometry.triangle_mesh import TriangleMesh
from ..plotting import draw_markers


@jaxtyped(typechecker=typechecker)
class TriangleScene(eqx.Module):
    """
    A simple scene made of triangle meshes, transmitters and receivers.

    Args:
        transmitters: The array of transmitter vertices.
        receivers: The array of receiver vertices.
        meshes: The list of triangle meshes.
    """

    transmitters: Float[Array, "num_transmitters 3"] = eqx.field(
        converter=jnp.asarray, default_factory=lambda: jnp.empty((0, 3))
    )
    """The array of transmitter vertices."""
    receivers: Float[Array, "num_receivers 3"] = eqx.field(
        converter=jnp.asarray, default_factory=lambda: jnp.empty((0, 3))
    )
    """The array of receiver vertices."""
    meshes: list[TriangleMesh] = eqx.field(default_factory=list)
    """The list of triangle meshes."""

    @classmethod
    def load_xml(cls, file: str) -> "TriangleScene":
        """
        Load a triangle scene from a XML file.

        This method uses
        :meth:`SionnaScene.load_xml<differt.scene.sionna.SionnaScene.load_xml>`
        internally.

        Args:
            file: The path to the XML file.

        Return:
            The corresponding scene containing only triangle meshes.
        """
        scene = _core.scene.triangle_scene.TriangleScene.load_xml(file)

        meshes = [
            TriangleMesh(vertices=mesh.vertices, triangles=mesh.triangles)
            for mesh in scene.meshes.values()
        ]
        return cls(meshes=meshes)

    def plot(
        self,
        tx_kwargs: Optional[Mapping[str, Any]] = None,
        rx_kwargs: Optional[Mapping[str, Any]] = None,
        meshes_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Plot this scene on a 3D scene.

        Args:
            tx_kwargs: A mutable mapping of keyword arguments passed to
                :py:func:`draw_markers<differt.plotting.draw_markers>`.
            rx_kwargs: A mutable mapping of keyword arguments passed to
                :py:func:`draw_markers<differt.plotting.draw_markers>`.
            meshes_kwargs: A mutable mapping of keyword arguments passed to
                :py:meth:`TriangleMesh.plot<differt.geometry.triangle_mesh.TriangleMesh.plot>`.
            kwargs: Keyword arguments passed to both
                :py:func:`draw_markers<differt.plotting.draw_markers>` and
                :py:meth:`TriangleMesh.plot<differt.geometry.triangle_mesh.TriangleMesh.plot>`.

        Return:
            The resulting plot output.
        """
        if tx_kwargs is None:
            tx_kwargs = {}

        if rx_kwargs is None:
            rx_kwargs = {}

        if meshes_kwargs is None:
            meshes_kwargs = {}

        result = None

        if self.transmitters.size > 0:
            result = draw_markers(np.asarray(self.transmitters), **tx_kwargs, **kwargs)

        if self.receivers.size > 0:
            result = draw_markers(np.asarray(self.receivers), **rx_kwargs, **kwargs)

        for mesh in self.meshes:
            result = mesh.plot(**meshes_kwargs, **kwargs)

        return result
