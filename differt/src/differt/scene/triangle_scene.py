"""Scene made of triangles and utilities."""

from collections.abc import Mapping
from typing import Any, Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

import differt_core.scene.triangle_scene
from differt.geometry.triangle_mesh import TriangleMesh
from differt.geometry.paths import Paths
from differt.plotting import draw_markers, reuse
from differt.rt.image_method import (
    consecutive_vertices_are_on_same_side_of_mirrors,
    image_method,
)
from differt.rt.utils import generate_all_path_candidates, rays_intersect_triangles
from differt.geometry.triangle_mesh import (
    triangles_contain_vertices_assuming_inside_same_plane,
)

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

    transmitters: Float[Array, "*transmitters_batch 3"] = eqx.field(
        converter=jnp.asarray,
        default_factory=lambda: jnp.empty((0, 3)),
    )
    """The array of transmitter vertices."""
    receivers: Float[Array, "*receivers_batch 3"] = eqx.field(
        converter=jnp.asarray,
        default_factory=lambda: jnp.empty((0, 3)),
    )
    """The array of receiver vertices."""
    mesh: TriangleMesh = eqx.field(default_factory=TriangleMesh.empty)
    """The triangle mesh."""

    @classmethod
    def from_core(
        cls, core_scene: differt_core.scene.triangle_scene.TriangleScene
    ) -> "TriangleScene":
        """
        Return a triangle scene from a scene created by the :mod:`differt_core` module.

        Args:
            core_scene: The scene from the core module.

        Returns:
            The corresponding scene.
        """
        return cls(
            mesh=TriangleMesh.from_core(core_scene.mesh),
        )

    @classmethod
    def load_xml(cls, file: str) -> "TriangleScene":
        """
        Load a triangle scene from a XML file.

        This method uses
        :meth:`SionnaScene.load_xml<differt_core.scene.sionna.SionnaScene.load_xml>`
        internally.

        Args:
            file: The path to the XML file.

        Returns:
            The corresponding scene containing only triangle meshes.
        """
        core_scene = differt_core.scene.triangle_scene.TriangleScene.load_xml(file)
        return cls.from_core(core_scene)

    @eqx.filter_jit
    def compute_paths(self, order: int) -> Paths:
        num_triangles = self.mesh.triangles.shape[0]
        path_candidates = generate_all_path_candidates(num_triangles, order)
        num_path_candidates = path_candidates.shape[0]
        tx_batch = self.transmitters.shape[:-1]
        rx_batch = self.receivers.shape[:-1]

        # [num_path_candidates *tx_batch *rx_batch 3]
        tx = jnp.expand_dims(self.transmitters, (0, *(-i for i, _ in enumerate(tx_batch, start=2))))
        rx = jnp.expand_dims(self.receivers, (0, *(i for i, _ in enumerate(rx_batch, start=1))))
        print(f"{tx.shape = }, {rx.shape = }")
        from_vertices = jnp.tile(tx, (num_path_candidates, *(1 for _ in tx_batch), *rx_batch, 1))
        to_vertices = jnp.tile(rx, (num_path_candidates, *tx_batch, *(1 for _ in rx_batch), 1))

        # [num_path_candidates order 3]
        triangles = jnp.take(self.mesh.triangles, path_candidates, axis=0)

        # [num_path_candidates order 3 3]
        triangle_vertices = jnp.take(self.mesh.vertices, triangles, axis=0)

        # [num_path_candidates order 3]
        mirror_vertices = triangle_vertices[
            ...,
            0,
            :,
        ]  # Only one vertex per triangle is needed
        # [num_path_candidates order 3]
        mirror_normals = jnp.take(self.mesh.normals, path_candidates, axis=0)

        # [num_path_candidates *tx_batch *rx_batch order 3]
        triangle_vertices = jnp.tile(triangle_vertices, (1, *tx_batch, *rx_batch, 1, 1, 1))
        mirror_vertices = jnp.tile(mirror_vertices, (1, *tx_batch, *rx_batch, 1, 1))
        mirror_normals = jnp.tile(mirror_normals, (1, *tx_batch, *rx_batch, 1, 1))

        # 2 - Trace paths

        # [num_path_candidates *tx_batch *rx_batch order 3]
        paths = image_method(from_vertices, to_vertices, mirror_vertices, mirror_normals)

        # 3 - Remove invalid paths

        # 3.1 - Remove paths with vertices outside triangles
        # [num_path_candidates *tx_batch *rx_batch order]
        mask = triangles_contain_vertices_assuming_inside_same_plane(
            triangle_vertices,
            paths,
        )
        # [num_path_candidates *tx_batch *rx_batch]
        mask = jnp.all(mask, axis=-1)

        # [num_path_candidates *tx_batch *rx_batch order+2 3]
        full_paths = jnp.concatenate(
            (
            jnp.expand_dims(from_vertices, axis=-2),
            paths,
            jnp.expand_dims(to_vertices, axis=-2),
            ),
            axis=-2,
        )

        # 3.2 - Remove paths with vertices not on the same side of mirrors
        # [num_path_candidates *tx_batch *rx_batch order]
        mask = consecutive_vertices_are_on_same_side_of_mirrors(
            full_paths,
            mirror_vertices,
            mirror_normals,
        )

        # [num_paths_inter]
        mask = jnp.all(mask, axis=-1)  # We will actually remove them later

        # 3.3 - Remove paths that are obstructed by other objects
        # [num_paths_inter order+1 3]
        ray_origins = full_paths[..., :-1, :]
        # [num_paths_inter order+1 3]
        ray_directions = jnp.diff(full_paths, axis=-2)

        # [num_paths_inter order+1 num_triangles 3]
        ray_origins = jnp.repeat(
            jnp.expand_dims(ray_origins, axis=-2),
            num_triangles,
            axis=-2,
        )
        # [num_paths_inter order+1 num_triangles 3]
        ray_directions = jnp.repeat(
            jnp.expand_dims(ray_directions, axis=-2),
            num_triangles,
            axis=-2,
        )

        # [num_paths_inter order+1 num_triangles], [num_paths_inter order+1 num_triangles]
        t, hit = rays_intersect_triangles(
            ray_origins,
            ray_directions,
            jnp.broadcast_to(all_triangle_vertices, (*ray_origins.shape, 3)),
        )
        # In theory, we could do t < 1.0 (because t == 1.0 means we are perfectly on a surface,
        # which is probably desirable, e.g., from a reflection) but in practice numerical
        # errors accumulate and will make this check impossible.
        # [num_paths_inter order+1 num_triangles]
        intersect = (t < 0.999) & hit
        #  [num_paths_inter]
        intersect = jnp.any(intersect, axis=(-1, -2))
        #  [num_paths_inter]
        mask = mask & ~intersect

        # 4 - Obtain final valid paths

        #  [num_paths_final]
        full_paths = full_paths[mask, ...]

        vertices = full_paths
        placeholders = - jnp.ones_like(path_candidates, shape=(*path_candidates.shape[:-1], 1))
        objects = jnp.concatenate((placeholders, path_candidates, placeholders), axis=-1)

        return Paths(vertices, objects)

    def plot(
        self,
        tx_kwargs: Optional[Mapping[str, Any]] = None,
        rx_kwargs: Optional[Mapping[str, Any]] = None,
        mesh_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:  # TODO: change output type
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

        Returns:
            The resulting plot output.
        """
        # TODO: remove **kwargs because reuse should already passed **kwargs.
        tx_kwargs = {"labels": "tx", **(tx_kwargs or {}), **kwargs}
        rx_kwargs = {"labels": "rx", **(rx_kwargs or {}), **kwargs}
        mesh_kwargs = {**(mesh_kwargs or {}), **kwargs}

        with reuse(**kwargs) as result:
            if self.transmitters.size > 0:
                draw_markers(np.asarray(self.transmitters).reshape((-1, 3)), **tx_kwargs)

            if self.receivers.size > 0:
                draw_markers(np.asarray(self.receivers).reshape((-1, 3)), **rx_kwargs)

            self.mesh.plot(**mesh_kwargs)

        return result
