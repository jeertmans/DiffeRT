"""Scene made of triangles and utilities."""

from collections.abc import Mapping
from functools import cached_property
from typing import Any, Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

import differt_core.scene.triangle_scene
from differt_core.scene.sionna import Material

from ..geometry.triangle_mesh import TriangleMesh
from ..plotting import draw_markers, reuse


@jaxtyped(typechecker=typechecker)
class TriangleScene(eqx.Module):
    """
    A simple scene made of one or more triangle meshes, some transmitters and some receivers.

    Args:
        transmitters: The array of transmitter vertices.
        receivers: The array of receiver vertices.
        meshes: The triangle mesh.
        materials: The mesh materials.
    """

    transmitters: Float[Array, "num_transmitters 3"] = eqx.field(
        converter=jnp.asarray, default_factory=lambda: jnp.empty((0, 3))
    )
    """The array of transmitter vertices."""
    receivers: Float[Array, "num_receivers 3"] = eqx.field(
        converter=jnp.asarray, default_factory=lambda: jnp.empty((0, 3))
    )
    """The array of receiver vertices."""
    meshes: tuple[TriangleMesh, ...] = eqx.field(converter=tuple, default_factory=tuple)
    """The triangle meshes."""
    materials: tuple[Material, ...] = eqx.field(converter=tuple, default_factory=tuple)
    """The mesh materials"""

    @cached_property
    def one_mesh(self) -> TriangleMesh:
        """
        Return a mesh that it the result of concatenating all meshes.

        This is especially useful for plotting, as plotting one large
        mesh is much faster than plotting many small ones.

        Return:
            The mesh that contains all meshes of this scene.
        """
        vertices = jnp.empty((0, 3))
        triangles = jnp.empty((0, 3), dtype=jnp.uint32)

        for mesh in self.meshes:
            offset = vertices.shape[0]
            vertices = jnp.concatenate((vertices, mesh.vertices))
            triangles = jnp.concatenate((triangles, mesh.triangles + offset))

        return TriangleMesh(vertices=vertices, triangles=triangles)

    @cached_property
    def face_colors(self) -> Float[Array, "num_triangles 3"]:
        """
        Return a (flattened) array of face colors, one for each triangle in each mesh.

        This is especially useful for plotting, and it to be used
        with :meth:`one_mesh`.

        Return:
            The mesh that contains all meshes of this scene.
        """
        colors = jnp.empty((0, 3))

        for mesh, material in zip(self.meshes, self.materials):
            num_triangles = mesh.triangles.shape[0]
            color = jnp.asarray(material.rgb)
            colors = jnp.concatenate((colors, jnp.tile(color, (num_triangles, 1))))

        return colors

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

        meshes = map(
            lambda mesh: TriangleMesh(
                vertices=mesh.vertices, triangles=mesh.triangles
            ),
            scene.meshes,
        )

        return cls(
            meshes=meshes,  # type: ignore[reportArgumentType]
            materials=scene.materials,
        )

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
        mesh_kwargs = {**(mesh_kwargs or {}), "face_colors": self.face_colors, **kwargs}

        with reuse(**kwargs) as result:
            if self.transmitters.size > 0:
                draw_markers(np.asarray(self.transmitters), **tx_kwargs)

            if self.receivers.size > 0:
                draw_markers(np.asarray(self.receivers), **rx_kwargs)

            self.one_mesh.plot(**mesh_kwargs)

        return result
