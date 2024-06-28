"""Mesh geometry made of triangles and utilities."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

import differt_core.scene.triangle_scene

from ..geometry.triangle_mesh import TriangleMesh
from ..plotting import draw_markers, reuse


@jaxtyped(typechecker=typechecker)
@dataclass  # TODO: see if we can still use eqx.Module (mutability issue)
class TriangleScene:
    """
    A simple scene made of one triangle mesh, some transmitters and some receivers.

    The triangle mesh can be the result of multiple triangle meshes being concatenated.

    Args:
        transmitters: The array of transmitter vertices.
        receivers: The array of receiver vertices.
        mesh: The triangle mesh.
        mesh_ids: A mapping between mesh IDs and the corresponding slice
            in :py:data:`mesh.triangles`.
    """

    transmitters: Float[Array, "num_transmitters 3"] = eqx.field(
        converter=jnp.asarray, default_factory=lambda: jnp.empty((0, 3))
    )
    """The array of transmitter vertices."""
    receivers: Float[Array, "num_receivers 3"] = eqx.field(
        converter=jnp.asarray, default_factory=lambda: jnp.empty((0, 3))
    )
    """The array of receiver vertices."""
    mesh: TriangleMesh = eqx.field(default_factory=TriangleMesh.empty)
    """The triangle mesh."""
    mesh_ids: dict[str, slice] = eqx.field(default_factory=dict)
    """A mapping between mesh IDs and the corresponding slice in :py:data:`mesh.triangles`."""

    @classmethod
    def load_xml(cls, file: str) -> "TriangleScene":
        """
        Load a triangle scene from a XML file.

        This method uses
        :meth:`SionnaScene.load_xml<differt_core.scene.sionna.SionnaScene.load_xml>`
        internally.

        Args:
            file: The path to the XML file.

        Return:
            The corresponding scene containing only triangle meshes.
        """
        scene = differt_core.scene.triangle_scene.TriangleScene.load_xml(file)

        mesh = TriangleMesh(
            vertices=scene.mesh.vertices, triangles=scene.mesh.triangles
        )
        return cls(mesh=mesh, mesh_ids=scene.mesh_ids)

    def plot(
        self,
        tx_kwargs: Optional[Mapping[str, Any]] = None,
        rx_kwargs: Optional[Mapping[str, Any]] = None,
        mesh_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Plot this scene on a 3D scene.

        Args:
            tx_kwargs: A mapping of keyword arguments passed to
                :py:func:`draw_markers<differt.plotting.draw_markers>`.
            rx_kwargs: A mapping of keyword arguments passed to
                :py:func:`draw_markers<differt.plotting.draw_markers>`.
            mesh_kwargs: A mapping of keyword arguments passed to
                :py:meth:`TriangleMesh.plot<differt.geometry.triangle_mesh.TriangleMesh.plot>`.
            kwargs: Keyword arguments passed to both
                :py:func:`draw_markers<differt.plotting.draw_markers>` and
                :py:meth:`TriangleMesh.plot<differt.geometry.triangle_mesh.TriangleMesh.plot>`.

        Return:
            The resulting plot output.
        """
        tx_kwargs = {"labels": "tx", **(tx_kwargs or {}), **kwargs}
        rx_kwargs = {"labels": "rx", **(rx_kwargs or {}), **kwargs}
        # TODO: add shapes color using BSDF
        mesh_kwargs = {**(mesh_kwargs or {}), **kwargs}

        with reuse(**kwargs) as result:
            if self.transmitters.size > 0:
                draw_markers(np.asarray(self.transmitters), **tx_kwargs)

            if self.receivers.size > 0:
                draw_markers(np.asarray(self.receivers), **rx_kwargs)

            self.mesh.plot(**mesh_kwargs)

        return result
