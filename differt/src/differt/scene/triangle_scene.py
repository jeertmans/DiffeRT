"""Scene made of triangles and utilities."""
# ruff: noqa: ERA001

from collections.abc import Iterator, Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Int, jaxtyped

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
from differt.rt.utils import (
    generate_all_path_candidates,
    generate_all_path_candidates_chunks_iter,
    rays_intersect_any_triangle,
)


@jaxtyped(typechecker=typechecker)
class TriangleScene(eqx.Module):
    """A simple scene made of one or more triangle meshes, some transmitters and some receivers."""

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
    @jax.jit
    @jaxtyped(typechecker=typechecker)
    def num_transmitters(self) -> int:
        """The number of transmitters."""
        return self.transmitters[..., 0].size

    @property
    @jax.jit
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

    def compute_paths(
        self, order: int, *, chunk_size: int | None = None, **kwargs: Any
    ) -> Paths | Iterator[Paths]:
        """
        Compute paths between all pairs of transmitters and receivers in the scene, that undergo a fixed number of interaction with objects.

        Args:
            order: The number of interaction, i.e., the number of bounces.
            chunk_size: If specified, it will iterate through chunks of path
                candidates, and yield the result as an iterator over paths chunks.
            kwargs: Keyword arguments passed to
                :func:`rays_intersect_any_triangle<differt.rt.utils.rays_intersect_any_triangle>`.

        Returns:
            The paths, as class wrapping path vertices, object indices, and a masked
            identify valid paths.
        """
        # 0 - Constants arrays of chunks
        num_triangles = self.mesh.triangles.shape[0]
        tx_batch = self.transmitters.shape[:-1]
        rx_batch = self.receivers.shape[:-1]

        # [tx_batch_flattened 3]
        from_vertices = self.transmitters.reshape(-1, 3)
        # [rx_batch_flattened 3]
        to_vertices = self.receivers.reshape(-1, 3)

        def _compute_paths(
            path_candidates: Int[Array, "num_path_candidates order"],
        ) -> Paths:
            # 1 - Broadcast arrays

            num_path_candidates = path_candidates.shape[0]

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
                **kwargs,
            ).any(axis=-1)  # Reduce on 'order'

            mask_3 = ~intersect

            # 4 - Generate output paths and reshape

            vertices = full_paths
            mask = mask_1 & mask_2 & mask_3

            # TODO: we also need to somehow mask degenerate paths, e.g., when two reflections occur on an edge

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

            objects = jnp.concatenate(
                (tx_objects, path_candidates, rx_objects), axis=-1
            )

            batch = (*tx_batch, *rx_batch, num_path_candidates)

            return Paths(
                vertices.reshape(*batch, order + 2, 3),
                objects.reshape(*batch, order + 2),
                mask.reshape(*batch),
            )

        if chunk_size:
            return (
                _compute_paths(path_candidates)
                for path_candidates in generate_all_path_candidates_chunks_iter(
                    num_triangles, order, chunk_size=chunk_size
                )
            )
        path_candidates = generate_all_path_candidates(num_triangles, order)
        return _compute_paths(path_candidates)

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
