"""Scene made of triangles and utilities."""
# ruff: noqa: ERA001

from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from beartype import beartype as typechecker
from jaxtyping import Array, ArrayLike, Float, jaxtyped

import differt_core.scene.triangle_scene
from differt.geometry.paths import Paths
from differt.geometry.triangle_mesh import (
    TriangleMesh,
    triangles_contain_vertices_assuming_inside_same_plane,
)
from differt.plotting import draw_markers, reuse
from differt.rt.image_method import (
    consecutive_vertices_are_on_same_side_of_mirrors,
    image_method,
)
from differt.rt.utils import generate_all_path_candidates, rays_intersect_triangles


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
    def compute_paths(self, order: int, hit_tol: Float[ArrayLike, ""] = 1e-3) -> Paths:
        """
        Compute paths between all pairs of transmitters and receivers in the scene, that undergo a fixed number of interaction with objects.

        Args:
            order: The number of interaction, i.e., the number of bounces.
            hit_tol: The tolerance applied to check if a ray hits another object or not,
                before it reaches the expected position, i.e., the 'interaction' object.

                Using a non-zero tolerance is required as it would otherwise trigger
                false positives.

        Returns:
            The paths, as class wrapping path vertices, object indices, and a masked
            identify valid paths.
        """
        # 1 - Broadcast arrays

        num_triangles = self.mesh.triangles.shape[0]
        path_candidates = generate_all_path_candidates(num_triangles, order)
        num_path_candidates = path_candidates.shape[0]
        tx_batch = self.transmitters.shape[:-1]
        rx_batch = self.receivers.shape[:-1]

        # [num_path_candidates *tx_batch *rx_batch 3]
        tx = jnp.expand_dims(
            self.transmitters, (0, *(-i for i, _ in enumerate(rx_batch, start=2)))
        )
        rx = jnp.expand_dims(
            self.receivers, (0, *(i for i, _ in enumerate(tx_batch, start=1)))
        )
        from_vertices = jnp.tile(
            tx, (num_path_candidates, *(1 for _ in tx_batch), *rx_batch, 1)
        )
        to_vertices = jnp.tile(
            rx, (num_path_candidates, *tx_batch, *(1 for _ in rx_batch), 1)
        )

        # [num_path_candidates order 3]
        triangles = jnp.take(self.mesh.triangles, path_candidates, axis=0)

        # [num_path_candidates order 3 3]
        triangle_vertices = jnp.take(self.mesh.vertices, triangles, axis=0)

        # [num_paths_candidates *tx_batch *rx_batch order 3 3]
        triangle_vertices = jnp.expand_dims(
            triangle_vertices,
            tuple(i for i, _ in enumerate((*tx_batch, *rx_batch), start=1)),
        )
        triangle_vertices = jnp.tile(
            triangle_vertices, (1, *tx_batch, *rx_batch, 1, 1, 1)
        )

        # [num_path_candidates *tx_batch *rx_batch order 3]
        mirror_vertices = triangle_vertices[
            ...,
            0,
            :,
        ]  # Only one vertex per triangle is needed

        # [num_path_candidates order 3]
        mirror_normals = jnp.take(self.mesh.normals, path_candidates, axis=0)

        # [num_path_candidates *tx_batch *rx_batch order 3]
        mirror_normals = jnp.expand_dims(
            mirror_normals,
            tuple(i for i, _ in enumerate((*tx_batch, *rx_batch), start=1)),
        )
        mirror_normals = jnp.tile(mirror_normals, (1, *tx_batch, *rx_batch, 1, 1))

        # 2 - Trace paths

        # [num_path_candidates *tx_batch *rx_batch order 3]
        paths = image_method(
            from_vertices, to_vertices, mirror_vertices, mirror_normals
        )

        # 3 - Identify invalid paths

        # 3.1 - Identify paths with vertices outside triangles
        # [num_path_candidates *tx_batch *rx_batch order]
        mask_1 = triangles_contain_vertices_assuming_inside_same_plane(
            triangle_vertices,
            paths,
        )
        # [num_path_candidates *tx_batch *rx_batch]
        mask_1 = jnp.all(mask_1, axis=-1)

        # [num_path_candidates *tx_batch *rx_batch order+2 3]
        full_paths = jnp.concatenate(
            (
                jnp.expand_dims(from_vertices, axis=-2),
                paths,
                jnp.expand_dims(to_vertices, axis=-2),
            ),
            axis=-2,
        )

        # 3.2 - Identify paths with vertices not on the same side of mirrors
        # [num_path_candidates *tx_batch *rx_batch order]
        mask_2 = consecutive_vertices_are_on_same_side_of_mirrors(
            full_paths,
            mirror_vertices,
            mirror_normals,
        )

        # [num_path_candidates *tx_batch *rx_batch]
        mask_2 = jnp.all(mask_2, axis=-1)  # We will actually remove them later

        # 3.3 - Identify paths that are obstructed by other objects
        # [num_path_candidates *tx_batch *rx_batch order+1 3]
        ray_origins = full_paths[..., :-1, :]
        # [num_path_candidates *tx_batch *rx_batch order+1 3]
        ray_directions = jnp.diff(full_paths, axis=-2)

        # [num_path_candidates *tx_batch *rx_batch order+1 num_triangles],
        # [num_path_candidates *tx_batch *rx_batch order+1 num_triangles]
        t, hit = rays_intersect_triangles(
            ray_origins,
            ray_directions,
            self.mesh.triangle_vertices,
        )
        # In theory, we could do t < 1.0 (because t == 1.0 means we are perfectly on a surface,
        # which is probably desirable, e.g., from a reflection) but in practice numerical
        # errors accumulate and will make this check impossible.
        # [num_path_candidates *tx_batch *rx_batch order+1 num_triangles]
        intersect = (t < (1.0 - hit_tol)) & hit
        #  [num_path_candidates *tx_batch *rx_batch]
        intersect = jnp.any(intersect, axis=(-1, -2))
        #  [num_path_candidates *tx_batch *rx_batch]
        mask_3 = ~intersect

        # 4 - Obtain final valid paths

        vertices = full_paths
        path_candidates = jnp.expand_dims(
            path_candidates,
            axis=tuple(i for i, _ in enumerate((*tx_batch, *rx_batch), start=1)),
        )
        placeholders = -jnp.ones_like(  # TODO: add transmitter/receiver indices if relevant?
            path_candidates, shape=(*path_candidates.shape[:-1], 1)
        )
        objects = jnp.concatenate(
            (placeholders, path_candidates, placeholders), axis=-1
        )
        objects = jnp.tile(objects, (1, *tx_batch, *rx_batch, 1))
        mask = mask_1 & mask_2 & mask_3

        return Paths(vertices, objects, mask)

    def plot(
        self,
        tx_kwargs: Mapping[str, Any] | None = None,
        rx_kwargs: Mapping[str, Any] | None = None,
        mesh_kwargs: Mapping[str, Any] | None = None,
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
                draw_markers(
                    np.asarray(self.transmitters).reshape((-1, 3)), **tx_kwargs
                )

            if self.receivers.size > 0:
                draw_markers(np.asarray(self.receivers).reshape((-1, 3)), **rx_kwargs)

            self.mesh.plot(**mesh_kwargs)

        return result
