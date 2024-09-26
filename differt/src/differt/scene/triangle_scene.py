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
from differt.geometry.utils import assemble_paths
from differt.plotting import draw_markers, reuse
from differt.rt.image_method import (
    consecutive_vertices_are_on_same_side_of_mirrors,
    image_method,
)
from differt.rt.utils import generate_all_path_candidates, rays_intersect_any_triangle


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

    @property
    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def num_transmitters(self) -> int:
        """The number of transmitters."""
        return self.transmitters[..., 0].size

    @property
    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def num_receivers(self) -> int:
        """The number of receivers."""
        return self.receivers[..., 0].size

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

    # @eqx.filter_jit
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

        # [tx_batch_flattened 3]
        from_vertices = self.transmitters.reshape(-1, 3)
        # [rx_batch_flattened 3]
        to_vertices = self.receivers.reshape(-1, 3)

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

        # 2 - Trace paths

        # [tx_batch_flat rx_batch_flat num_path_candidates order 3]
        paths = image_method(
            from_vertices[:, None, None, :],
            to_vertices[None, :, None, :],
            mirror_vertices,
            mirror_normals,
        )

        # 3 - Identify invalid paths

        # 3.1 - Identify paths with vertices outside triangles
        # [tx_batch_flat rx_batch_flat num_path_candidates]
        mask_1 = triangles_contain_vertices_assuming_inside_same_plane(
            triangle_vertices,
            paths,
        ).all(axis=-1)  # Reduce on 'order'

        # [tx_batch_flat rx_batch_flat num_path_candidates order+2 3]
        full_paths = assemble_paths(
            from_vertices[:, None, None, None, :],
            paths,
            to_vertices[None, :, None, None, :],
        )

        # 3.2 - Identify paths with vertices not on the same side of mirrors
        # [tx_batch_flat rx_batch_flat num_path_candidates]
        mask_2 = consecutive_vertices_are_on_same_side_of_mirrors(
            full_paths,
            mirror_vertices,
            mirror_normals,
        ).all(axis=-1)  # Reduce on 'order'

        # 3.3 - Identify paths that are obstructed by other objects
        # [tx_batch_flat rx_batch_flat num_path_candidates order+1 3]
        ray_origins = full_paths[..., :-1, :]
        # [tx_batch_flat rx_batch_flat num_path_candidates order+1 3]
        ray_directions = jnp.diff(full_paths, axis=-2)

        # [tx_batch_flat rx_batch_flat num_path_candidates]
        intersect = rays_intersect_any_triangle(
            ray_origins,
            ray_directions,
            self.mesh.triangle_vertices,
            hit_threshold=(1.0 - hit_tol),
        ).any(axis=-1)  # Reduce on 'order'

        mask_3 = ~intersect

        # 4 - Generate output paths and reshape

        vertices = full_paths
        mask = mask_1 & mask_2 & mask_3

        object_dtype = path_candidates.dtype

        tx_objects = jnp.arange(self.num_transmitters, dtype=object_dtype)
        rx_objects = jnp.arange(self.num_receivers, dtype=object_dtype)

        tx_objects = jnp.broadcast_to(
            tx_objects,
            (self.num_transmitters, self.num_receivers, num_path_candidates, 1),
        )
        rx_objects = jnp.broadcast_to(
            tx_objects,
            (self.num_transmitters, self.num_receivers, num_path_candidates, 1),
        )
        path_candidates = jnp.broadcast_to(
            path_candidates,
            (self.num_transmitters, self.num_receivers, num_path_candidates, order),
        )

        objects = jnp.concatenate((tx_objects, path_candidates, rx_objects), axis=-1)

        batch = (*tx_batch, *rx_batch, num_path_candidates)

        return Paths(
            vertices.reshape(*batch, order + 2, 3),
            objects.reshape(*batch, order + 2),
            mask.reshape(*batch),
        )

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
                :func:`draw_markers<differt.plotting.draw_markers>`.
            rx_kwargs: A mapping of keyword arguments passed to
                :func:`draw_markers<differt.plotting.draw_markers>`.
            mesh_kwargs: A mapping of keyword arguments passed to
                :meth:`TriangleMesh.plot<differt.geometry.triangle_mesh.TriangleMesh.plot>`.
            kwargs: Keyword arguments passed to
                :func:`reuse<differt.plotting.reuse>`.

        Returns:
            The resulting plot output.
        """
        # TODO: remove **kwargs because reuse should already passed **kwargs.
        tx_kwargs = {"labels": "tx", **(tx_kwargs or {})}
        rx_kwargs = {"labels": "rx", **(rx_kwargs or {})}
        mesh_kwargs = {} if mesh_kwargs is None else mesh_kwargs

        with reuse(**kwargs) as result:
            if self.transmitters.size > 0:
                draw_markers(
                    np.asarray(self.transmitters).reshape((-1, 3)), **tx_kwargs
                )

            if self.receivers.size > 0:
                draw_markers(np.asarray(self.receivers).reshape((-1, 3)), **rx_kwargs)

            self.mesh.plot(**mesh_kwargs)

        return result
