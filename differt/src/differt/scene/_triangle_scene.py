import math
import typing
import warnings
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, Literal, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, Int

import differt_core.scene
from differt.geometry import (
    Paths,
    SBRPaths,
    TriangleMesh,
    assemble_paths,
    fibonacci_lattice,
    viewing_frustum,
)
from differt.plotting import PlotOutput, draw_markers, reuse
from differt.rt import (
    SizedIterator,
    consecutive_vertices_are_on_same_side_of_mirrors,
    first_triangles_hit_by_rays,
    image_method,
    rays_intersect_any_triangle,
    rays_intersect_triangles,
    triangles_visible_from_vertices,
)
from differt.utils import smoothing_function
from differt_core.rt import CompleteGraph, DiGraph

if TYPE_CHECKING or hasattr(typing, "GENERATING_DOCS"):
    from typing import Self

    try:
        from sionna.rt import Scene as SionnaScene
    except ImportError:
        SionnaScene = Any
else:
    Self = Any  # Because runtime type checking from 'beartype' will fail when combined with 'jaxtyping'
    SionnaScene = Any


@eqx.filter_jit
def _compute_paths(
    mesh: TriangleMesh,
    tx_vertices: Float[Array, "num_tx_vertices 3"],
    rx_vertices: Float[Array, "num_rx_vertices 3"],
    path_candidates: Int[Array, "num_path_candidates order"],
    *,
    epsilon: Float[ArrayLike, " "] | None,
    hit_tol: Float[ArrayLike, " "] | None,
    min_len: Float[ArrayLike, " "] | None,
    smoothing_factor: Float[ArrayLike, " "] | None,
    confidence_threshold: Float[ArrayLike, " "] = 0.5,
    batch_size: int | None,
) -> Paths:
    if min_len is None:
        dtype = jnp.result_type(mesh.vertices, tx_vertices, tx_vertices)
        min_len = 10 * jnp.finfo(dtype).eps

    min_len = jnp.asarray(min_len)

    # 1 - Broadcast arrays

    num_tx_vertices = tx_vertices.shape[0]
    num_rx_vertices = rx_vertices.shape[0]
    num_path_candidates, order = path_candidates.shape

    if mesh.assume_quads:
        # [num_path_candidates 2*order]
        path_candidates = jnp.repeat(path_candidates, 2, axis=-1)
        path_candidates = path_candidates.at[..., 1::2].add(1)  # Shift odd indices by 1
        k = 2
    else:
        k = 1

    # [num_path_candidates k*order 3]
    triangles = jnp.take(mesh.triangles, path_candidates, axis=0).reshape(
        num_path_candidates, k * order, 3
    )  # reshape required if mesh is empty

    # [num_path_candidates k*order 3 3]
    triangle_vertices = jnp.take(mesh.vertices, triangles, axis=0).reshape(
        num_path_candidates, k * order, 3, 3
    )  # reshape required if mesh is empty

    if mesh.mask is not None:
        # For a ray to be active, it must hit triangles that are not masked out (i.e, inactive).
        # [num_path_candidates]  # noqa: ERA001
        active_rays = jnp.take(mesh.mask, path_candidates, axis=0).all(axis=-1)
    else:
        active_rays = None

    # [num_path_candidates order 3]
    mirror_vertices = triangle_vertices[
        ...,
        :: (2 if mesh.assume_quads else 1),
        0,
        :,
    ]  # Only one vertex per triangle is needed

    # [num_path_candidates order 3]
    mirror_normals = jnp.take(
        mesh.normals, path_candidates[..., :: (2 if mesh.assume_quads else 1)], axis=0
    )

    # 2 - Trace paths

    if num_path_candidates == 0:
        dtype = jnp.result_type(
            tx_vertices, rx_vertices, mirror_vertices, mesh.vertices
        )
        # [num_tx_vertices num_rx_vertices num_path_candidates order+2 3]
        full_paths = jnp.empty(
            (num_tx_vertices, num_rx_vertices, 0, order + 2, 3), dtype=dtype
        )
    else:
        # [num_tx_vertices num_rx_vertices num_path_candidates order 3]
        paths = image_method(
            tx_vertices[:, None, None, :],
            rx_vertices[None, :, None, :],
            mirror_vertices,
            mirror_normals,
        )
        full_paths = assemble_paths(
            tx_vertices[:, None, None, :],
            paths,
            rx_vertices[None, :, None, :],
        )

    # 3 - Identify invalid paths

    # [num_tx_vertices num_rx_vertices num_path_candidates order+1 3]
    ray_origins = full_paths[..., :-1, :]
    # [num_tx_vertices num_rx_vertices num_path_candidates order+1 3]
    ray_directions = jnp.diff(full_paths, axis=-2)

    # 3.1 - Check if paths vertices are inside respective triangles

    # [num_tx_vertices num_rx_vertices num_path_candidates]
    if mesh.assume_quads:
        if smoothing_factor is not None:
            inside_triangles = (
                rays_intersect_triangles(
                    jnp.repeat(ray_origins[..., :-1, :], 2, axis=-2),
                    jnp.repeat(ray_directions[..., :-1, :], 2, axis=-2),
                    triangle_vertices,
                    epsilon=epsilon,
                    smoothing_factor=smoothing_factor,
                )[1]
                .reshape(
                    num_tx_vertices, num_rx_vertices, num_path_candidates, order, 2
                )
                .max(axis=-1, initial=0.0)
                .min(axis=-1, initial=1.0)
            )  # Reduce on 'order' axis and on the two triangles (per quad)
        else:
            inside_triangles = (
                rays_intersect_triangles(
                    jnp.repeat(ray_origins[..., :-1, :], 2, axis=-2),
                    jnp.repeat(ray_directions[..., :-1, :], 2, axis=-2),
                    triangle_vertices,
                    epsilon=epsilon,
                )[1]
                .reshape(
                    num_tx_vertices, num_rx_vertices, num_path_candidates, order, 2
                )
                .any(axis=-1)
                .all(axis=-1)
            )  # Reduce on 'order' axis and on the two triangles (per quad)
    elif smoothing_factor is not None:
        inside_triangles = rays_intersect_triangles(
            ray_origins[..., :-1, :],
            ray_directions[..., :-1, :],
            triangle_vertices,
            epsilon=epsilon,
            smoothing_factor=smoothing_factor,
        )[1].min(axis=-1, initial=1.0)  # Reduce on 'order' axis
    else:
        inside_triangles = rays_intersect_triangles(
            ray_origins[..., :-1, :],
            ray_directions[..., :-1, :],
            triangle_vertices,
            epsilon=epsilon,
        )[1].all(axis=-1)  # Reduce on 'order' axis

    # 3.2 - Check if consecutive path vertices are on the same side of mirrors

    # [num_tx_vertices num_rx_vertices num_path_candidates]
    if smoothing_factor is not None:
        valid_reflections = consecutive_vertices_are_on_same_side_of_mirrors(
            full_paths,
            mirror_vertices,
            mirror_normals,
            smoothing_factor=smoothing_factor,
        ).min(axis=-1, initial=1.0)  # Reduce on 'order'
    else:
        valid_reflections = consecutive_vertices_are_on_same_side_of_mirrors(
            full_paths,
            mirror_vertices,
            mirror_normals,
        ).all(axis=-1)  # Reduce on 'order'

    # 3.3 - Identify paths that are blocked by other objects

    # [num_tx_vertices num_rx_vertices num_path_candidates]
    if smoothing_factor is not None:
        blocked = rays_intersect_any_triangle(
            ray_origins,
            ray_directions,
            mesh.triangle_vertices,
            active_triangles=mesh.mask,
            epsilon=epsilon,
            hit_tol=hit_tol,
            smoothing_factor=smoothing_factor,
            batch_size=batch_size,
        ).max(axis=-1, initial=0.0)  # Reduce on 'order'
    else:
        blocked = rays_intersect_any_triangle(
            ray_origins,
            ray_directions,
            mesh.triangle_vertices,
            active_triangles=mesh.mask,
            epsilon=epsilon,
            hit_tol=hit_tol,
            batch_size=batch_size,
        ).any(axis=-1)  # Reduce on 'order'

    # 3.4 - Identify path segments that are too small (e.g., double-reflection inside an edge)

    ray_lengths = jnp.sum(ray_directions * ray_directions, axis=-1)  # Squared norm

    if smoothing_factor is not None:
        too_small = smoothing_function(min_len - ray_lengths, smoothing_factor).max(
            axis=-1, initial=0.0
        )  # Any path segment being too small
    else:
        too_small = (ray_lengths < min_len).any(
            axis=-1
        )  # Any path segment being too small

    # 3.5 - Identify paths that are not finite
    is_finite = jnp.isfinite(full_paths).all(axis=(-1, -2))
    full_paths = jnp.where(
        is_finite[..., None, None], full_paths, jnp.zeros_like(full_paths)
    )

    if smoothing_factor is not None:
        mask = None
        confidence = jnp.stack(
            (
                inside_triangles,
                valid_reflections,
                1.0 - blocked,
                1.0 - too_small,
                is_finite.astype(inside_triangles.dtype),
            ),
            axis=-1,
        ).min(axis=-1, initial=1.0)
        if active_rays is not None:
            confidence *= active_rays
    else:
        confidence = None
        mask = inside_triangles & valid_reflections & ~blocked & ~too_small & is_finite
        if active_rays is not None:
            mask &= active_rays

    vertices = full_paths

    # 4 - Generate output paths and reshape

    object_dtype = path_candidates.dtype

    tx_objects = jnp.arange(num_tx_vertices, dtype=object_dtype)
    rx_objects = jnp.arange(num_rx_vertices, dtype=object_dtype)

    tx_objects = jnp.broadcast_to(
        tx_objects[:, None, None, None],
        (num_tx_vertices, num_rx_vertices, num_path_candidates, 1),
    )
    rx_objects = jnp.broadcast_to(
        rx_objects[None, :, None, None],
        (num_tx_vertices, num_rx_vertices, num_path_candidates, 1),
    )
    path_candidates = jnp.broadcast_to(
        path_candidates[:, ::k],
        (
            num_tx_vertices,
            num_rx_vertices,
            num_path_candidates,
            order,
        ),
    )

    objects = jnp.concatenate((tx_objects, path_candidates, rx_objects), axis=-1)

    return Paths(
        vertices,
        objects,
        mask=mask,
        confidence=confidence,
        confidence_threshold=confidence_threshold,
    )


@eqx.filter_jit
def _compute_paths_sbr(
    mesh: TriangleMesh,
    tx_vertices: Float[Array, "num_tx_vertices 3"],
    rx_vertices: Float[Array, "num_rx_vertices 3"],
    *,
    order: int,
    num_rays: int,
    epsilon: Float[ArrayLike, " "] | None,
    max_dist: Float[ArrayLike, " "],
    batch_size: int | None,
) -> SBRPaths:
    # 1 - Prepare arrays

    # [num_triangles 3 3]
    triangle_vertices = mesh.triangle_vertices

    num_tx_vertices = tx_vertices.shape[0]
    num_rx_vertices = rx_vertices.shape[0]

    world_vertices = jnp.concatenate(
        (triangle_vertices.reshape(-1, 3), rx_vertices), axis=0
    )

    # [num_tx_vertices 2 3]
    # TODO: handle mesh.mask
    frustums = jax.vmap(viewing_frustum, in_axes=(0, None))(tx_vertices, world_vertices)

    # [num_tx_vertices num_rays 2 3]
    ray_origins = jnp.broadcast_to(
        tx_vertices[:, None, :], (num_tx_vertices, num_rays, 3)
    )
    ray_directions = jax.vmap(
        lambda frustum: fibonacci_lattice(num_rays, frustum=frustum)
    )(frustums)

    def scan_fun(
        ray_origins_directions_and_valids: tuple[
            Float[Array, "num_tx_vertices num_rays 3"],
            Float[Array, "num_tx_vertices num_rays 3"],
            Bool[Array, "num_tx_vertices num_rays"],
        ],
        _: None,
    ) -> tuple[
        tuple[
            Float[Array, "num_tx_vertices num_rays 3"],
            Float[Array, "num_tx_vertices num_rays 3"],
            Bool[Array, "num_tx_vertices num_rays"],
        ],
        tuple[
            Int[Array, "num_tx_vertices num_rays"],
            Float[Array, "num_tx_vertices num_rays 3"],
            Bool[Array, "num_tx_vertices num_rx_vertices num_rays"],
        ],
    ]:
        # [num_tx_vertices num_rays 3],
        # [num_tx_vertices num_rays 3],
        # [num_tx_vertices num_rays]
        (
            ray_origins,
            ray_directions,
            valid_rays,
        ) = ray_origins_directions_and_valids

        # 1 - Compute next intersection with triangles

        # [num_tx_vertices num_rays]
        triangles, t_hit = first_triangles_hit_by_rays(
            ray_origins,
            ray_directions,
            triangle_vertices,
            active_triangles=mesh.mask,
            epsilon=epsilon,
            batch_size=batch_size,
        )

        # 2 - Check if the rays pass near RX

        # [num_tx_vertices num_rx_vertices num_rays 3]
        ray_origins_to_rx_vertices = (
            rx_vertices[None, :, None, :] - ray_origins[:, None, ...]
        )

        # [num_tx_vertices num_rx_vertices num_rays]
        ray_distances_to_rx_vertices = jnp.square(
            jnp.cross(ray_directions[:, None, ...], ray_origins_to_rx_vertices)
        ).sum(axis=-1)  # Squared distance from rays to RXs

        # [num_tx_vertices num_rx_vertices num_rays]
        t_rxs = jnp.sum(
            ray_directions[:, None, ...] * ray_origins_to_rx_vertices, axis=-1
        )  # Distance (scaled by ray directions) from RXs projected onto rays to ray origins

        masks = jnp.where(
            (t_rxs > 0) & (t_rxs < t_hit[:, None, :]) & valid_rays[:, None, :],
            ray_distances_to_rx_vertices < max_dist,
            False,  # noqa: FBT003
        )

        # 3 - Update rays

        # [num_tx_vertices num_rays 3]
        mirror_normals = jnp.take(mesh.normals, triangles, axis=0)

        # Mark rays leaving the scene as invalid
        inside_scene = jnp.isfinite(t_hit)
        valid_rays &= inside_scene
        # And avoid creating NaNs
        t_hit = jnp.where(inside_scene, t_hit, jnp.zeros_like(t_hit))

        ray_origins += t_hit[..., None] * ray_directions
        ray_directions = (
            ray_directions
            - 2.0
            * jnp.sum(ray_directions * mirror_normals, axis=-1, keepdims=True)
            * mirror_normals
        )

        return (ray_origins, ray_directions, valid_rays), (
            triangles,
            ray_origins,
            masks,
        )

    valid_rays = jnp.ones(ray_origins.shape[:-1], dtype=bool)
    _, (path_candidates, vertices, masks) = jax.lax.scan(
        scan_fun,
        (ray_origins, ray_directions, valid_rays),
        length=order + 1,
    )

    path_candidates = jnp.moveaxis(path_candidates[:-1, ...], 0, -1)
    vertices = jnp.moveaxis(vertices[:-1, ...], 0, -2)
    masks = jnp.moveaxis(masks, 0, -1)

    # 4 - Generate output paths and reshape

    vertices = assemble_paths(
        tx_vertices[:, None, None, :],
        vertices[:, None, ...],  # We already excluded last vertex
        rx_vertices[None, :, None, :],  # And replace it with receiver vertices
    )

    object_dtype = path_candidates.dtype

    tx_objects = jnp.arange(num_tx_vertices, dtype=object_dtype)
    rx_objects = jnp.arange(num_rx_vertices, dtype=object_dtype)

    tx_objects = jnp.broadcast_to(
        tx_objects[:, None, None, None],
        (num_tx_vertices, num_rx_vertices, num_rays, 1),
    )
    rx_objects = jnp.broadcast_to(
        rx_objects[None, :, None, None],
        (num_tx_vertices, num_rx_vertices, num_rays, 1),
    )
    path_candidates = jnp.broadcast_to(
        path_candidates[:, None, ...],
        (
            num_tx_vertices,
            num_rx_vertices,
            num_rays,
            order,
        ),
    )

    objects = jnp.concatenate((tx_objects, path_candidates, rx_objects), axis=-1)

    return SBRPaths(
        vertices,
        objects,
        masks=masks,
    )


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
    def num_transmitters(self) -> int:
        """The number of transmitters."""
        return math.prod(self.transmitters.shape[:-1])

    @property
    def num_receivers(self) -> int:
        """The number of receivers."""
        return math.prod(self.receivers.shape[:-1])

    def set_assume_quads(self, flag: bool = True) -> Self:
        """
        Return a new instance of this scene with :attr:`TriangleMesh.assume_quads<differt.geometry.TriangleMesh.assume_quads>` set to ``flag``.

        This is simply a convenient wrapper to call :meth:`TriangleMesh.set_assume_quads<differt.geometry.TriangleMesh.set_assume_quads>` on the inner :attr:`mesh` attribute.

        Args:
            flag: The new flag value.

        Returns:
            A new scene with the same structure with the inner mesh's :attr:`TriangleMesh.assume_quads<differt.geometry.TriangleMesh.assume_quads>` set to ``flag``.
        """
        return eqx.tree_at(lambda s: s.mesh, self, self.mesh.set_assume_quads(flag))

    def with_transmitters_grid(
        self, m: int = 50, n: int | None = 50, *, height: Float[ArrayLike, " "] = 1.5
    ) -> Self:
        """
        Return a new instance of this scene with a 2D grid of transmitters placed at a fixed height.

        The transmitters are uniformly spaced on the whole scene.

        Args:
            m: The number of sample along x dimension.
            n: The number of sample along y dimension,
                defaults to ``m`` is left unspecified.
            height: The height at which transmitters are placed.

        Returns:
            The new scene with a 2D grid of transmitters.
        """
        if n is None:
            n = m

        dtype = self.mesh.vertices.dtype

        (min_x, min_y, _), (max_x, max_y, _) = self.mesh.bounding_box

        x, y = jnp.meshgrid(
            jnp.linspace(min_x, max_x, m, dtype=dtype),
            jnp.linspace(min_y, max_y, n, dtype=dtype),
        )
        z = jnp.full_like(x, height)

        return eqx.tree_at(
            lambda s: s.transmitters, self, jnp.stack((x, y, z), axis=-1)
        )

    def with_receivers_grid(
        self, m: int = 50, n: int | None = 50, *, height: Float[ArrayLike, " "] = 1.5
    ) -> Self:
        """
        Return a new instance of this scene with a 2D grid of receivers placed at a fixed height.

        The receivers are uniformly spaced on the whole scene.

        Args:
            m: The number of sample along x dimension.
            n: The number of sample along y dimension,
                defaults to ``m`` is left unspecified.
            height: The height at which receivers are placed.

        Returns:
            The new scene with a 2D grid of receivers.
        """
        if n is None:
            n = m

        dtype = self.mesh.vertices.dtype

        (min_x, min_y, _), (max_x, max_y, _) = self.mesh.bounding_box

        x, y = jnp.meshgrid(
            jnp.linspace(min_x, max_x, m, dtype=dtype),
            jnp.linspace(min_y, max_y, n, dtype=dtype),
        )
        z = jnp.full_like(x, height)

        return eqx.tree_at(lambda s: s.receivers, self, jnp.stack((x, y, z), axis=-1))

    def rotate(self, rotation_matrix: Float[ArrayLike, "3 3"]) -> Self:
        """
        Return a new scene by applying a rotation matrix to all the objects in the scene.

        Args:
            rotation_matrix: The rotation matrix.

        Returns:
            The new rotated scene.
        """
        rotation_matrix = jnp.asarray(rotation_matrix)
        return eqx.tree_at(
            lambda s: (s.transmitters, s.receivers, s.mesh),
            self,
            (
                (rotation_matrix @ self.transmitters.reshape(-1, 3).T).T.reshape(
                    self.transmitters.shape
                ),
                (rotation_matrix @ self.receivers.reshape(-1, 3).T).T.reshape(
                    self.receivers.shape
                ),
                self.mesh.rotate(rotation_matrix),
            ),
        )

    @eqx.filter_jit
    def scale(self, scale_factor: Float[ArrayLike, " "]) -> Self:
        """
        Return a new scene by applying a scale factor to all the objects in the scene.

        Args:
            scale_factor: The scale factor.

        Returns:
            The new scaled scene.
        """
        scale_factor = jnp.asarray(scale_factor)
        return eqx.tree_at(
            lambda s: (s.transmitters, s.receivers, s.mesh),
            self,
            (
                self.transmitters * scale_factor,
                self.receivers * scale_factor,
                self.mesh.scale(scale_factor),
            ),
        )

    def translate(self, translation: Float[ArrayLike, "3"]) -> Self:
        """
        Return a new scene by applying a translation to all the objects in the scene.

        Args:
            translation: The translation vector.

        Returns:
            The new translated scene.
        """
        translation = jnp.asarray(translation)
        return eqx.tree_at(
            lambda s: (s.transmitters, s.receivers, s.mesh),
            self,
            (
                self.transmitters + translation,
                self.receivers + translation,
                self.mesh.translate(translation),
            ),
        )

    @classmethod
    def from_core(cls, core_scene: differt_core.scene.TriangleScene) -> Self:
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
    def load_xml(cls, file: str) -> Self:
        """
        Load a triangle scene from a XML file.

        This method uses
        :meth:`SionnaScene.load_xml<differt_core.scene.SionnaScene.load_xml>`
        internally.

        Args:
            file: The path to the XML file.

        Returns:
            The corresponding scene containing only triangle meshes.
        """
        core_scene = differt_core.scene.TriangleScene.load_xml(file)
        return cls.from_core(core_scene)

    @classmethod
    def from_mitsuba(cls, mi_scene) -> Self:  # noqa: ANN001  # for some reason, mi.Scene cannot be imported, but only supports delayed annotations, which is not compatible with jaxtyping
        """
        Load a triangle scene from a Mitsuba scene object.

        This method does not extract any transmitters or receivers from the Mitsuba scene,
        as Mitsuba does not provide any explicit information about them, and they are usually
        part of the Sionna scene object, see :meth:`from_sionna`.

        Args:
            mi_scene (mitsuba.Scene): The Mitsuba scene object.

                You can obtain the Mitsuba scene object from a Sionna scene
                its ``.mi_scene`` attribute.

        Returns:
            The corresponding scene containing only triangle meshes.

        .. seealso::

            :meth:`from_sionna`
        """
        mesh = TriangleMesh.empty()

        for shape in mi_scene.shapes():
            rm = shape.bsdf().radio_material
            mesh += (
                TriangleMesh(
                    vertices=shape.vertex_positions_buffer().jax().reshape(-1, 3),
                    triangles=shape.faces_buffer().jax().reshape(-1, 3),
                )
                .set_face_colors(jnp.asarray(rm.color))
                .set_materials(f"itu_{rm.itu_type}")
                .set_face_materials(0)
            )

        return cls(
            mesh=mesh,
        )

    @classmethod
    def from_sionna(cls, sionna_scene: SionnaScene) -> Self:  # type: ignore[reportUndefinedVariable]
        """
        Load a triangle scene from a Sionna scene object.

        This method uses :meth:`from_mitsuba` internally to load the scene objects.

        .. warning::
            Using this method is only recommended if you already have a Sionna scene object.
            Otherwise, you can use :meth:`load_xml` to load a scene from a XML file, compatible with Sionna,
            at a faster speed.

        .. warning::
            This method does not *currently* use any information about possible antenna arrays.

        Args:
            sionna_scene: The Sionna scene object.

        Returns:
            The corresponding scene containing only triangle meshes.
        """
        scene = cls.from_mitsuba(sionna_scene.mi_scene)

        return eqx.tree_at(
            lambda s: (s.transmitters, s.receivers),
            scene,
            (
                jnp.concatenate([
                    tx.position.jax().reshape(1, 3)
                    for tx in sionna_scene.transmitters.values()
                ])
                if sionna_scene.transmitters
                else jnp.empty((0, 3)),
                jnp.concatenate([
                    rx.position.jax().reshape(1, 3)
                    for rx in sionna_scene.receivers.values()
                ])
                if sionna_scene.receivers
                else jnp.empty((0, 3)),
            ),
        )

    @overload
    def compute_paths(
        self,
        order: int | None = None,
        *,
        method: Literal["exhaustive"] = "exhaustive",
        chunk_size: None = None,
        num_rays: int = int(1e6),
        path_candidates: Int[ArrayLike, "num_path_candidates order"] | None = None,
        epsilon: Float[ArrayLike, " "] | None = None,
        hit_tol: Float[ArrayLike, " "] | None = None,
        min_len: Float[ArrayLike, " "] | None = None,
        max_dist: Float[ArrayLike, " "] = 1e-3,
        smoothing_factor: Float[ArrayLike, " "] | None = None,
        confidence_threshold: Float[ArrayLike, " "] = 0.5,
        batch_size: int | None = 512,
        disconnect_inactive_triangles: bool = False,
    ) -> Paths: ...

    @overload
    def compute_paths(
        self,
        order: int,
        *,
        method: Literal["hybrid"],
        chunk_size: None = None,
        num_rays: int = int(1e6),
        path_candidates: None = None,
        epsilon: Float[ArrayLike, " "] | None = None,
        hit_tol: Float[ArrayLike, " "] | None = None,
        min_len: Float[ArrayLike, " "] | None = None,
        max_dist: Float[ArrayLike, " "] = 1e-3,
        smoothing_factor: Float[ArrayLike, " "] | None = None,
        confidence_threshold: Float[ArrayLike, " "] = 0.5,
        batch_size: int | None = 512,
        disconnect_inactive_triangles: bool = False,
    ) -> Paths: ...

    @overload
    def compute_paths(
        self,
        order: int | None = None,
        *,
        method: Literal["exhaustive"] = "exhaustive",
        chunk_size: int,
        num_rays: int = int(1e6),
        path_candidates: None = None,
        epsilon: Float[ArrayLike, " "] | None = None,
        hit_tol: Float[ArrayLike, " "] | None = None,
        min_len: Float[ArrayLike, " "] | None = None,
        max_dist: Float[ArrayLike, " "] = 1e-3,
        smoothing_factor: Float[ArrayLike, " "] | None = None,
        confidence_threshold: Float[ArrayLike, " "] = 0.5,
        batch_size: int | None = 512,
        disconnect_inactive_triangles: bool = False,
    ) -> SizedIterator[Paths]: ...

    @overload
    def compute_paths(
        self,
        order: int,
        *,
        method: Literal["hybrid"],
        chunk_size: int,
        num_rays: int = int(1e6),
        path_candidates: None = None,
        epsilon: Float[ArrayLike, " "] | None = None,
        hit_tol: Float[ArrayLike, " "] | None = None,
        min_len: Float[ArrayLike, " "] | None = None,
        max_dist: Float[ArrayLike, " "] = 1e-3,
        smoothing_factor: Float[ArrayLike, " "] | None = None,
        confidence_threshold: Float[ArrayLike, " "] = 0.5,
        batch_size: int | None = 512,
        disconnect_inactive_triangles: bool = False,
    ) -> Iterator[Paths]: ...

    @overload
    def compute_paths(
        self,
        order: int | None = None,
        *,
        method: Literal["exhaustive"] = "exhaustive",
        chunk_size: int,
        num_rays: int = int(1e6),
        path_candidates: Int[ArrayLike, "num_path_candidates order"],
        epsilon: Float[ArrayLike, " "] | None = None,
        hit_tol: Float[ArrayLike, " "] | None = None,
        min_len: Float[ArrayLike, " "] | None = None,
        max_dist: Float[ArrayLike, " "] = 1e-3,
        smoothing_factor: Float[ArrayLike, " "] | None = None,
        confidence_threshold: Float[ArrayLike, " "] = 0.5,
        batch_size: int | None = 512,
        disconnect_inactive_triangles: bool = False,
    ) -> Paths: ...

    @overload
    def compute_paths(
        self,
        order: int,
        *,
        method: Literal["sbr"],
        chunk_size: None = None,
        num_rays: int = int(1e6),
        path_candidates: None = None,
        epsilon: Float[ArrayLike, " "] | None = None,
        hit_tol: None = None,
        min_len: None = None,
        max_dist: Float[ArrayLike, " "] = 1e-3,
        smoothing_factor: None = None,
        confidence_threshold: Float[ArrayLike, " "] = 0.5,
        batch_size: int | None = 512,
        disconnect_inactive_triangles: bool = False,
    ) -> SBRPaths: ...

    def compute_paths(
        self,
        order: int | None = None,
        *,
        method: Literal["exhaustive", "sbr", "hybrid"] = "exhaustive",
        chunk_size: int | None = None,
        num_rays: int = int(1e6),
        path_candidates: Int[ArrayLike, "num_path_candidates order"] | None = None,
        epsilon: Float[ArrayLike, " "] | None = None,
        hit_tol: Float[ArrayLike, " "] | None = None,
        min_len: Float[ArrayLike, " "] | None = None,
        max_dist: Float[ArrayLike, " "] = 1e-3,
        smoothing_factor: Float[ArrayLike, " "] | None = None,
        confidence_threshold: Float[ArrayLike, " "] = 0.5,
        batch_size: int | None = 512,
        disconnect_inactive_triangles: bool = False,
    ) -> Paths | SizedIterator[Paths] | Iterator[Paths] | SBRPaths:
        """
        Compute paths between all pairs of transmitters and receivers in the scene, that undergo a fixed number of interaction with objects.

        Note:
            Currently, only :abbr:`LOS (line of sight)` and fixed ``order`` reflection paths are computed,
            using the :func:`image_method<differt.rt.image_method>`. More types of interactions
            and path tracing methods will be added in the future, so stay tuned!

        Args:
            order: The number of interaction, i.e., the number of bounces.

                This or ``path_candidates`` must be specified.
            method: The method used to generate path candidates and trace paths.

                See :ref:`advanced_path_tracing` for a detailed tutorial.

                * If ``'exhaustive'``, all possible paths are generated, performing
                  an exhaustive search. This is the slowest method, but it is also
                  the most accurate.
                * If ``'sbr'``, a fixed number of rays are launched from each transmitter
                  and are allowed to perform a fixed number of bounces. Only rays paths
                  passing in the vicinity of a receiver are considered valid, see
                  ``max_dist`` parameter. This is the fastest method, but may miss
                  some valid paths if the number of rays is too low.

                  .. important::

                    This method is currently unstable and not yet optimized, and
                    it is likely to changed in future releases. Use with caution.
                * If ``'hybrid'``, a hybrid method is used, which estimates the objects
                  visible from all transmitters, to reduce the number of path candidates,
                  by launching a fixed number of rays, and then performs an exhaustive
                  search on those path candidates. This is a faster alternative to
                  ``'exhaustive'``, but still grows exponentially with the number of
                  bounces or the size of the scene. In the future, we plan on allowing
                  the user to explicitly pass visibility matrices to further reduce the
                  number of path candidates.

                  .. warning::
                    This method is best used for a single transmitter and a single receiver,
                    as the estimated visibility is merged across all transmitters and receivers,
                    respectively.

            chunk_size: If specified, it will iterate through chunks of path
                candidates, and yield the result as an iterator over paths chunks.

                Unused if ``path_candidates`` is provided or if ``method == 'sbr'``.
            num_rays: The number of rays launched with ``method == 'sbr'`` or
                ``method == 'hybrid'``.

                Unused if ``method == 'exhaustive'``.
            path_candidates: An optional array of path candidates, see :ref:`path_candidates`.

                This is helpful to only generate paths on a subset of the scene. E.g., this
                is used in :ref:`sampling-paths` to test a specific set of path candidates
                generated from a Machine Learning model.

                If :attr:`self.mesh.assume_quads<differt.geometry.TriangleMesh.assume_quads>`
                is :data:`True`, then path candidates are
                rounded down toward the nearest even value (but object indices still refer
                to triangle indices, not quadrilateral indices).

                **Not compatible with** ``method == 'sbr'`` and ``method == 'hybrid'``.
            epsilon: Tolerance for checking ray / objects intersection, see
                :func:`rays_intersect_triangles<differt.rt.rays_intersect_triangles>`.
            hit_tol: Tolerance for checking blockage (i.e., obstruction), see
                :func:`rays_intersect_any_triangle<differt.rt.rays_intersect_any_triangle>`.

                Unused if ``method == 'sbr'``.
            min_len: Minimal (squared [#f1]_) length that each path segment must have for a path to be valid.

                If not specified, the default is ten times the epsilon value
                of the currently used floating point dtype.

                Unused if ``method == 'sbr'``.

            max_dist: Maximal (squared [#f1]_) distance between a receiver and a ray for the receiver
                to be considered in the vicinity of the ray path.

                Unused if ``method == 'exhaustive'`` or if ``method == 'hybrid'``.
            smoothing_factor: If set, intermediate hard conditions are replaced with smoothed ones,
                as described in :cite:`fully-eucap2024`, and this argument parameters the slope
                of the smoothing function. The, valid paths are lazily identified using
                ``confidence > confidence_threshold`` where ``confidence`` is a real value
                between 0 and 1 that indicates the confidence that a path is valid.

                For more details, refer to :ref:`smoothing`.

                  .. warning::

                    Currently, only the ``'exhaustive'`` method is supported.
            confidence_threshold: A threshold value for deciding which paths are valid.
            batch_size: If specified, the number of triangles or rays to process in one batch
                when checking for intersections.

                If :data:`None`, everything is processed in one batch, which can lead to
                memory issues on large scenes.

                See :func:`rays_intersect_any_triangle<differt.rt.rays_intersect_any_triangle>`,
                :func:`triangles_visible_from_vertices<differt.rt.triangles_visible_from_vertices>`,
                and :func:`first_triangles_hit_by_rays<differt.rt.first_triangles_hit_by_rays>`
                for more details.
            disconnect_inactive_triangles: If :data:`True`, inactive triangles (where
                the mesh mask is :data:`False`) are disconnected from the graph before
                generating path candidates. This can significantly reduce computational
                time for scenes with many inactive triangles, but the path candidates
                array size will vary based on the mask, which can trigger recompilations
                in JIT-compiled code.

                For the ``'hybrid'`` method, inactive triangles are always disconnected
                regardless of this parameter value, as the method already depends on
                the mask.


        Returns:
            The paths, as class wrapping path vertices, object indices, and a masked
            identify valid paths.

            The returned paths have the following batch dimensions:

            * ``[*transmitters_batch *receivers_batch num_path_candidates]``,
            * ``[*transmitters_batch *receivers_batch chunk_size]``,
            * or ``[*transmitters_batch *receivers_batch num_rays]``,

            depending on the method used.

        Raises:
            ValueError: If neither ``order`` nor ``path_candidates`` has been provided,
                or if both have been provided simultaneously.

                If ``method == 'sbr'`` or ``method == 'hybrid'`, and ``order`` is not provided.

        .. [#f1] Passing the squared length/distance is useful to avoid computing square root values, which is expensive.
        """
        if (order is None) == (path_candidates is None):
            msg = "You must specify one of 'order' or `path_candidates`, not both."
            raise ValueError(msg)
        if (chunk_size is not None) and (path_candidates is not None):
            msg = "Argument 'chunk_size' is ignored when 'path_candidates' is provided."
            warnings.warn(msg, UserWarning, stacklevel=2)
            chunk_size = None
        if (method != "exhaustive") and (smoothing_factor is not None):
            msg = "Argument 'smoothing' is currently ignored when 'method' is not set to 'exhaustive'."
            warnings.warn(msg, UserWarning, stacklevel=2)
            smoothing_factor = None

        tx_batch = self.transmitters.shape[:-1]
        rx_batch = self.receivers.shape[:-1]

        if method == "sbr":
            if order is None:
                msg = "Argument 'order' is required when 'method == \"sbr\"'."
                raise ValueError(msg)

            return _compute_paths_sbr(
                self.mesh,
                self.transmitters.reshape(-1, 3),
                self.receivers.reshape(-1, 3),
                order=order,
                num_rays=num_rays,
                epsilon=epsilon,
                max_dist=max_dist,
                batch_size=batch_size,
            ).reshape(*tx_batch, *rx_batch, -1)

        # 0 - Constants arrays of chunks
        assume_quads = self.mesh.assume_quads

        # [tx_batch_flattened 3]
        tx_vertices = self.transmitters.reshape(-1, 3)
        # [rx_batch_flattened 3]
        rx_vertices = self.receivers.reshape(-1, 3)

        graph = CompleteGraph(self.mesh.num_primitives)

        if method == "hybrid":
            if order is None:
                msg = "Argument 'order' is required when 'method == \"hybrid\"'."
                raise ValueError(msg)

            triangles_visible_from_tx = triangles_visible_from_vertices(
                tx_vertices,
                self.mesh.triangle_vertices,
                active_triangles=self.mesh.mask,
                num_rays=num_rays,
                epsilon=epsilon,
                batch_size=batch_size,
            ).any(axis=0)  # reduce on all transmitters

            triangles_visible_from_rx = triangles_visible_from_vertices(
                rx_vertices,
                self.mesh.triangle_vertices,
                active_triangles=self.mesh.mask,
                num_rays=num_rays,
                epsilon=epsilon,
                batch_size=batch_size,
            ).any(axis=0)  # reduce on all receivers

            if assume_quads:
                triangles_visible_from_tx = triangles_visible_from_tx.reshape(
                    -1, 2
                ).any(axis=-1)  # seeing any triangle of a quad is enough
                triangles_visible_from_rx = triangles_visible_from_rx.reshape(
                    -1, 2
                ).any(axis=-1)  # seeing any triangle of a quad is enough

            graph = DiGraph.from_complete_graph(graph)
            from_, to = graph.insert_from_and_to_nodes(
                from_adjacency=np.asarray(triangles_visible_from_tx),
                to_adjacency=np.asarray(triangles_visible_from_rx),
            )
            if self.mesh.mask is not None:
                # The number of path candidates generated by the 'hybrid' method already
                # depends on the mask, so we will always disconnect nodes in that case.
                mask = self.mesh.mask
                if assume_quads:
                    # For quads, we need both triangles to be active
                    mask = mask[0::2] & mask[1::2]
                graph.filter_by_mask(
                    np.asarray(mask), fast_mode=True
                )  # Further reduce graph size by removing inactive triangles
        elif disconnect_inactive_triangles and self.mesh.mask is not None:
            mask = self.mesh.mask
            if assume_quads:
                # For quads, we need both triangles to be active
                mask = mask[0::2] & mask[1::2]

            graph = DiGraph.from_complete_graph(graph)
            from_, to = graph.insert_from_and_to_nodes()
            graph.filter_by_mask(np.asarray(mask), fast_mode=True)
        else:
            from_ = graph.num_nodes
            to = from_ + 1

        if chunk_size:
            path_candidates_iter = graph.all_paths_array_chunks(
                from_=from_,
                to=to,
                depth=order + 2,  # type: ignore[reportOptionalOperand]
                include_from_and_to=False,
                chunk_size=chunk_size,
            )
            it = (
                _compute_paths(
                    self.mesh,
                    tx_vertices,
                    rx_vertices,
                    jnp.asarray(
                        2 * path_candidates if assume_quads else path_candidates,
                        dtype=int,
                    ),
                    epsilon=epsilon,
                    hit_tol=hit_tol,
                    min_len=min_len,
                    smoothing_factor=smoothing_factor,
                    confidence_threshold=confidence_threshold,
                    batch_size=batch_size,
                ).reshape(*tx_batch, *rx_batch, path_candidates.shape[0])
                for path_candidates in path_candidates_iter
            )

            if hasattr(path_candidates_iter, "__len__"):
                return SizedIterator(it, size=path_candidates_iter.__len__)
            return it

        if path_candidates is None:
            path_candidates = jnp.asarray(
                graph.all_paths_array(
                    from_=from_,
                    to=to,
                    depth=order + 2,  # type: ignore[reportOptionalOperand]
                    include_from_and_to=False,
                ),
                dtype=int,
            )

            if self.mesh.assume_quads:
                path_candidates = 2 * path_candidates
        else:
            path_candidates = jnp.asarray(path_candidates)
            if self.mesh.assume_quads:
                path_candidates -= path_candidates % 2

        return _compute_paths(
            self.mesh,
            tx_vertices,
            rx_vertices,
            path_candidates,
            epsilon=epsilon,
            hit_tol=hit_tol,
            min_len=min_len,
            smoothing_factor=smoothing_factor,
            confidence_threshold=confidence_threshold,
            batch_size=batch_size,
        ).reshape(*tx_batch, *rx_batch, path_candidates.shape[0])

    def plot(
        self,
        tx_kwargs: Mapping[str, Any] | None = None,
        rx_kwargs: Mapping[str, Any] | None = None,
        mesh_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> PlotOutput:
        """
        Plot this scene on a 3D scene.

        Args:
            tx_kwargs: A mapping of keyword arguments passed to
                :func:`draw_markers<differt.plotting.draw_markers>`.
            rx_kwargs: A mapping of keyword arguments passed to
                :func:`draw_markers<differt.plotting.draw_markers>`.
            mesh_kwargs: A mapping of keyword arguments passed to
                :meth:`TriangleMesh.plot<differt.geometry.TriangleMesh.plot>`.
            kwargs: Keyword arguments passed to
                :func:`reuse<differt.plotting.reuse>`.

        Returns:
            The resulting plot output.
        """
        tx_kwargs = {"labels": "tx", **(tx_kwargs or {})}
        rx_kwargs = {"labels": "rx", **(rx_kwargs or {})}
        mesh_kwargs = {} if mesh_kwargs is None else mesh_kwargs

        with reuse(pass_all_kwargs=True, **kwargs) as result:
            if self.transmitters.size > 0:
                draw_markers(self.transmitters, **tx_kwargs)

            if self.receivers.size > 0:
                draw_markers(self.receivers, **rx_kwargs)

            self.mesh.plot(**mesh_kwargs)

        return result
