# ruff: noqa: ERA001

import math
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, overload

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
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
    generate_all_path_candidates,
    generate_all_path_candidates_chunks_iter,
    image_method,
    rays_intersect_any_triangle,
    rays_intersect_triangles,
    triangles_visible_from_vertices,
)
from differt.utils import dot

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self
else:
    Self = Any  # Because runtime type checking from 'beartype' will fail when combined with 'jaxtyping'


@eqx.filter_jit
def _compute_paths(
    mesh: TriangleMesh,
    tx_vertices: Float[Array, "num_tx_vertices 3"],
    rx_vertices: Float[Array, "num_rx_vertices 3"],
    path_candidates: Int[Array, "num_path_candidates order"],
    *,
    parallel: bool,
    epsilon: Float[ArrayLike, " "] | None,
    hit_tol: Float[ArrayLike, " "] | None,
    min_len: Float[ArrayLike, " "] | None,
) -> Paths:
    if min_len is None:
        dtype = jnp.result_type(mesh.vertices, tx_vertices, tx_vertices)
        min_len = 10 * jnp.finfo(dtype).eps

    # 1 - Broadcast arrays

    num_tx_vertices = tx_vertices.shape[0]
    num_rx_vertices = rx_vertices.shape[0]
    num_path_candidates, order = path_candidates.shape

    # [num_path_candidates order 3]
    triangles = jnp.take(mesh.triangles, path_candidates, axis=0).reshape(
        num_path_candidates, order, 3
    )  # reshape required if mesh is empty

    # [num_path_candidates order 3 3]
    triangle_vertices = jnp.take(mesh.vertices, triangles, axis=0).reshape(
        num_path_candidates, order, 3, 3
    )  # reshape required if mesh is empty

    if mesh.assume_quads:
        # [num_path_candidates order 2 3]
        quads = jnp.take(
            mesh.triangles,
            jnp.stack((path_candidates, path_candidates + 1), axis=-1),
            axis=0,
        ).reshape(num_path_candidates, order, 2, 3)  # reshape required if mesh is empty

        # [num_path_candidates order 2 3 3]
        quad_vertices = jnp.take(mesh.vertices, quads, axis=0).reshape(
            num_path_candidates, order, 2, 3, 3
        )  # reshape required if mesh is empty
    else:
        quad_vertices = None

    # [num_path_candidates order 3]
    mirror_vertices = triangle_vertices[
        ...,
        0,
        :,
    ]  # Only one vertex per triangle is needed

    # [num_path_candidates order 3]
    mirror_normals = jnp.take(mesh.normals, path_candidates, axis=0)

    def fun(
        tx_vertices: Float[Array, "num_tx_vertices 3"],
        rx_vertices: Float[Array, "num_rx_vertices 3"],
    ) -> tuple[
        Float[
            Array,
            "num_tx_vertices num_rx_vertices num_path_candidates path_length 3",
        ],
        Bool[Array, "num_tx_vertices num_rx_vertices num_path_candidates"],
    ]:
        # 2 - Trace paths

        # [num_tx_vertices num_rx_vertices num_path_candidates order 3]
        paths = image_method(
            tx_vertices[:, None, None, :],
            rx_vertices[None, :, None, :],
            mirror_vertices,
            mirror_normals,
        )

        # [num_tx_vertices num_rx_vertices num_path_candidates order+2 3]
        full_paths = assemble_paths(
            tx_vertices[:, None, None, None, :],
            paths,
            rx_vertices[None, :, None, None, :],
        )

        # 3 - Identify invalid paths

        # [num_tx_vertices num_rx_vertices num_path_candidates order+1 3]
        ray_origins = full_paths[..., :-1, :]
        # [num_tx_vertices num_rx_vertices num_path_candidates order+1 3]
        ray_directions = jnp.diff(full_paths, axis=-2)

        # 3.1 - Check if paths vertices are inside respective triangles

        # [num_tx_vertices num_rx_vertices num_path_candidates]
        if mesh.assume_quads:
            inside_triangles = (
                rays_intersect_triangles(
                    ray_origins[..., :-1, None, :],
                    ray_directions[..., :-1, None, :],
                    quad_vertices,  # type: ignore[reportArgumentType]
                    epsilon=epsilon,
                )[1]
                .any(axis=-1)
                .all(axis=-1)
            )  # Reduce on 'order' axis and on the two triangles (per quad)
        else:
            inside_triangles = rays_intersect_triangles(
                ray_origins[..., :-1, :],
                ray_directions[..., :-1, :],
                triangle_vertices,
                epsilon=epsilon,
            )[1].all(axis=-1)  # Reduce on 'order' axis

        # 3.2 - Check if consecutive path vertices are on the same side of mirrors

        # [num_tx_vertices num_rx_vertices num_path_candidates]
        valid_reflections = consecutive_vertices_are_on_same_side_of_mirrors(
            full_paths,
            mirror_vertices,
            mirror_normals,
        ).all(axis=-1)  # Reduce on 'order'

        # 3.3 - Identify paths that are blocked by other objects

        # [num_tx_vertices num_rx_vertices num_path_candidates]
        blocked = rays_intersect_any_triangle(
            ray_origins,
            ray_directions,
            mesh.triangle_vertices,
            epsilon=epsilon,
            hit_tol=hit_tol,
        ).any(axis=-1)  # Reduce on 'order'

        # 3.4 - Identify path segments that are too small (e.g., double-reflection inside an edge)

        ray_lengths = dot(ray_directions)  # Squared norm

        too_small = (ray_lengths < min_len).any(
            axis=-1
        )  # Any path segment being too small

        # TODO: check if we should invalidate non-finite paths
        # is_finite = jnp.isfinite(full_paths).all(axis=(-1, -2))

        mask = inside_triangles & valid_reflections & ~blocked & ~too_small

        return full_paths, mask

    if parallel:
        num_devices = jax.device_count()

        if (num_tx_vertices * num_rx_vertices) % num_devices == 0:
            tx_mesh = math.gcd(num_tx_vertices, num_devices)
            rx_mesh = num_devices // tx_mesh
            in_specs = (P("i", None), P("j", None))
            out_specs = (P("i", "j", None, None, None), P("i", "j", None))
        else:
            msg = (
                f"Found {num_devices} devices available, "
                "but could not find any input with a size that is a multiple of that value. "
                "Please user a number of transmitter and receiver points that is a "
                f"multiple of {num_devices}."
            )
            raise ValueError(msg)

        fun = shard_map(  # type: ignore[reportAssigmentType]
            fun,
            Mesh(
                mesh_utils.create_device_mesh((tx_mesh, rx_mesh)), axis_names=("i", "j")
            ),
            in_specs=in_specs,
            out_specs=out_specs,
        )

    vertices, mask = fun(tx_vertices, rx_vertices)

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
        path_candidates,
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
        mask,
    )


@eqx.filter_jit
def _compute_paths_sbr(
    mesh: TriangleMesh,
    tx_vertices: Float[Array, "num_tx_vertices 3"],
    rx_vertices: Float[Array, "num_rx_vertices 3"],
    *,
    order: int,
    num_rays: int,
    parallel: bool,
    epsilon: Float[ArrayLike, " "] | None,
    max_dist: Float[ArrayLike, " "],
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
            epsilon=epsilon,
        )

        # 2 - Check if the rays pass near RX

        # [num_tx_vertices num_rx_vertices num_rays]
        ray_origins_to_rx_vertices = (
            rx_vertices[None, :, None, :] - ray_origins[:, None, ...]
        )

        # [num_tx_vertices num_rx_vertices num_rays]
        ray_distances_to_rx_vertices = dot(
            jnp.cross(ray_directions[:, None, ...], ray_origins_to_rx_vertices)
        )  # Squared distance from rays to RXs

        # [num_tx_vertices num_rx_vertices num_rays]
        t_rxs = dot(
            ray_directions[:, None, ...], ray_origins_to_rx_vertices
        )  # Distance (scaled by ray directions) from RXs projected onto rays to ray origins

        masks = jnp.where(
            (t_rxs > 0) & (t_rxs < t_hit[:, None, :]) & valid_rays[:, None, :],
            ray_distances_to_rx_vertices < max_dist,
            False,  # noqa: FBT003
        )

        # 3 - Update rays

        # [num_tx_vertices num_rays 3]
        mirror_normals = jnp.take(mesh.normals, triangles, axis=0)

        ray_origins += t_hit[..., None] * ray_directions
        ray_directions = (
            ray_directions
            - 2.0 * dot(ray_directions, mirror_normals, keepdims=True) * mirror_normals
        )

        # Mark rays leaving the scene as invalid
        valid_rays &= jnp.isfinite(t_hit)

        return (ray_origins, ray_directions, valid_rays), (
            triangles,
            ray_origins,
            masks,
        )

    def fun(
        ray_origins: Float[Array, "num_tx_vertices num_rays 3"],
        ray_directions: Float[Array, "num_tx_vertices num_rays 3"],
    ) -> tuple[
        Int[Array, "order_plus_1 num_tx_vertices num_rays"],
        Float[Array, "order_plus_1 num_tx_vertices num_rays 3"],
        Bool[Array, "order_plus_1 num_tx_vertices num_rx_vertices num_rays"],
    ]:
        valid_rays = jnp.ones(ray_origins.shape[:-1], dtype=bool)
        _, (path_candidates, vertices, masks) = jax.lax.scan(
            scan_fun,
            (ray_origins, ray_directions, valid_rays),
            length=order + 1,
        )
        return (path_candidates, vertices, masks)

    if parallel:
        num_devices = jax.device_count()

        if (num_tx_vertices * num_rays) % num_devices == 0:
            tx_mesh = math.gcd(num_tx_vertices, num_devices)
            ray_mesh = num_devices // tx_mesh
            in_specs = (P("i", "j", None), P("i", "j", None))
            out_specs = (
                P(None, "i", "j"),
                P(None, "i", "j", None),
                P(None, "i", None, "j"),
            )
        else:
            msg = (
                f"Found {num_devices} devices available, "
                "but could not find any input with a size that is a multiple of that value. "
                "Please user a number of transmitters and rays that is a "
                f"multiple of {num_devices}."
            )
            raise ValueError(msg)

        fun = shard_map(  # type: ignore[reportAssigmentType]
            fun,
            Mesh(
                mesh_utils.create_device_mesh((tx_mesh, ray_mesh)),
                axis_names=("i", "j"),
            ),
            in_specs=in_specs,
            out_specs=out_specs,
        )

    path_candidates, vertices, masks = fun(ray_origins, ray_directions)

    path_candidates = jnp.moveaxis(path_candidates[:-1, ...], 0, -1)
    vertices = jnp.moveaxis(vertices[:-1, ...], 0, -2)
    masks = jnp.moveaxis(masks, 0, -1)

    # 4 - Generate output paths and reshape

    vertices = assemble_paths(
        tx_vertices[:, None, None, None, :],
        vertices[:, None, ...],  # We already excluded last vertex
        rx_vertices[None, :, None, None, :],  # And replace it with receiver vertices
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
        return self.transmitters[..., 0].size

    @property
    def num_receivers(self) -> int:
        """The number of receivers."""
        return self.receivers[..., 0].size

    def set_assume_quads(self, flag: bool = True) -> Self:
        """
        Return a copy of this scene with :attr:`TriangleMesh.assume_quads<differt.geometry.TriangleMesh.assume_quads>` set to ``flag``.

        This is simply a convenient wrapper around :meth:`TriangleMesh.set_assume_quads<differt.geometry.TriangleMesh.set_assume_quads>`.

        Args:
            flag: The new flag value.

        Returns:
            A new scene.
        """
        return eqx.tree_at(lambda s: s.mesh, self, self.mesh.set_assume_quads(flag))

    def with_transmitters_grid(
        self, m: int = 50, n: int | None = 50, *, height: Float[ArrayLike, " "] = 1.5
    ) -> Self:
        """
        Return a copy of this scene with a 2D grid of transmitters placed at a fixed height.

        The transmitters are uniformly spaced on the whole scene.

        Args:
            m: The number of sample along x dimension.
            n: The number of sample along y dimension,
                defaults to ``m`` is left unspecified.
            height: The height at which transmitters are placed.

        Returns:
            The new scene.
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
        Return a copy of this scene with a 2D grid of receivers placed at a fixed height.

        The receivers are uniformly spaced on the whole scene.

        Args:
            m: The number of sample along x dimension.
            n: The number of sample along y dimension,
                defaults to ``m`` is left unspecified.
            height: The height at which receivers are placed.

        Returns:
            The new scene.
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
            scale_factor: The scate factor.

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

    @overload
    def compute_paths(
        self,
        order: int | None = None,
        *,
        method: Literal["exhaustive", "hybrid"] = "exhaustive",
        chunk_size: None = None,
        num_rays: int = int(1e6),
        path_candidates: Int[ArrayLike, "num_path_candidates order"] | None = None,
        parallel: bool = False,
        epsilon: Float[ArrayLike, " "] | None = None,
        hit_tol: Float[ArrayLike, " "] | None = None,
        min_len: Float[ArrayLike, " "] | None = None,
        max_dist: Float[ArrayLike, " "] = 1e-3,
    ) -> Paths: ...

    @overload
    def compute_paths(
        self,
        order: int | None = None,
        *,
        method: Literal["exhaustive", "hybrid"] = "exhaustive",
        chunk_size: int,
        num_rays: int = int(1e6),
        path_candidates: None = None,
        parallel: bool = False,
        epsilon: Float[ArrayLike, " "] | None = None,
        hit_tol: Float[ArrayLike, " "] | None = None,
        min_len: Float[ArrayLike, " "] | None = None,
        max_dist: Float[ArrayLike, " "] = 1e-3,
    ) -> SizedIterator[Paths]: ...

    @overload
    def compute_paths(
        self,
        order: int | None = None,
        *,
        method: Literal["exhaustive"] = "exhaustive",
        chunk_size: int,
        num_rays: int = int(1e6),
        path_candidates: Int[ArrayLike, "num_path_candidates order"],
        parallel: bool = False,
        epsilon: Float[ArrayLike, " "] | None = None,
        hit_tol: Float[ArrayLike, " "] | None = None,
        min_len: Float[ArrayLike, " "] | None = None,
        max_dist: Float[ArrayLike, " "] = 1e-3,
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
        parallel: bool = False,
        epsilon: Float[ArrayLike, " "] | None = None,
        hit_tol: None = None,
        min_len: None = None,
        max_dist: Float[ArrayLike, " "] = 1e-3,
    ) -> SBRPaths: ...

    def compute_paths(  # noqa: C901
        self,
        order: int | None = None,
        *,
        method: Literal["exhaustive", "sbr", "hybrid"] = "exhaustive",
        chunk_size: int | None = None,
        num_rays: int = int(1e6),
        path_candidates: Int[ArrayLike, "num_path_candidates order"] | None = None,
        parallel: bool = False,
        epsilon: Float[ArrayLike, " "] | None = None,
        hit_tol: Float[ArrayLike, " "] | None = None,
        min_len: Float[ArrayLike, " "] | None = None,
        max_dist: Float[ArrayLike, " "] = 1e-3,
    ) -> Paths | SizedIterator[Paths] | SBRPaths:
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
                  by launching a fixed number of rays, and then performs an extausive
                  search on those path candidates. This is a faster alternative to
                  ``'exhaustive'``, but still grows exponentially with the number of
                  bounces or the size of the scene. In the future, we plan on allowing
                  the user to explicitly pass visibility matrices to further reduce the
                  number of path candidates.

                  .. warning::

                    The ``'hybrid'`` method is not yet implemented.

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

                **Not compatible with** ``method == 'sbr'``.
            parallel: If :data:`True`, ray tracing is performed in parallel across all available
                devices. The number of transmitters times the number of receivers
                **must** be a multiple of :func:`jax.device_count`, otherwise an error is raised.

                When ``method == 'sbr'``, the number of transmitters times the number of rays
                **must** be a multiple of :func:`jax.device_count`, otherwise an error is raised.
            epsilon: Tolelance for checking ray / objects intersection, see
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

                If ``method == 'sbr'``, ``order`` is required.

        .. [#f1] Passing the squared length/distance is useful to avoid computing square root values, which is expensive.
        """
        if (order is None) == (path_candidates is None):
            msg = "You must specify one of 'order' or `path_candidates`, not both."
            raise ValueError(msg)
        if (chunk_size is not None) and (path_candidates is not None):
            msg = "Argument 'chunk_size' is ignored when 'path_candidates' is provided."
            warnings.warn(msg, UserWarning, stacklevel=2)
            chunk_size = None

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
                parallel=parallel,
                epsilon=epsilon,
                max_dist=max_dist,
            ).reshape(*tx_batch, *rx_batch, -1)
        if method == "hybrid":
            msg = "Hybrid method not implemented yet."
            raise NotImplementedError(msg)  # TODO: implement
            visibility = triangles_visible_from_vertices(
                self.transmitters, self.mesh.triangle_vertices
            )

            if self.mesh.assume_quads:
                visibility = visibility.reshape(-1, 2).any(axis=-1)

        # 0 - Constants arrays of chunks
        num_objects = (
            self.mesh.num_quads if self.mesh.assume_quads else self.mesh.num_triangles
        )

        # [tx_batch_flattened 3]
        tx_vertices = self.transmitters.reshape(-1, 3)
        # [rx_batch_flattened 3]
        rx_vertices = self.receivers.reshape(-1, 3)

        if chunk_size:
            path_candidates_iter = generate_all_path_candidates_chunks_iter(
                num_objects,
                order,  # type: ignore[reportArgumentType]
                chunk_size=chunk_size,
            )
            size = path_candidates_iter.__len__
            it = (
                _compute_paths(
                    self.mesh,
                    tx_vertices,
                    rx_vertices,
                    2 * path_candidates if self.mesh.assume_quads else path_candidates,
                    parallel=parallel,
                    epsilon=epsilon,
                    hit_tol=hit_tol,
                    min_len=min_len,
                ).reshape(*tx_batch, *rx_batch, path_candidates.shape[0])
                for path_candidates in path_candidates_iter
            )

            return SizedIterator(it, size=size)

        if path_candidates is None:
            path_candidates = generate_all_path_candidates(
                num_objects,
                order,
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
            parallel=parallel,
            epsilon=epsilon,
            hit_tol=hit_tol,
            min_len=min_len,
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

        with reuse(**kwargs) as result:
            if self.transmitters.size > 0:
                draw_markers(self.transmitters, **tx_kwargs)

            if self.receivers.size > 0:
                draw_markers(self.receivers, **rx_kwargs)

            self.mesh.plot(**mesh_kwargs)

        return result
