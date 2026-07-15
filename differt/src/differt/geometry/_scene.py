import dataclasses
import math
import typing
import warnings
from collections.abc import Callable, Iterator, Mapping
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Unpack,
    cast,
    no_type_check,
    overload,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from jaxtyping import Array, ArrayLike, Float, Int
from jaxtyping import UInt as Uint

import differt_core.geometry
from differt.plotting import PlotOutput, draw_markers, reuse

from ._mesh import Mesh
from ._paths import LaunchedPaths, TracedPaths
from ._solvers import (
    AbstractPathLauncher,
    AbstractPathTracer,
    ExhaustivePathTracer,
    HybridPathTracer,
    SBRPathLauncher,
    _ExhaustivePathTracerKwargs,
    _HybridPathTracerKwargs,
    _SBRPathLauncherKwargs,
)
from ._utils import SizedIterator, fibonacci_lattice, viewing_frustum

if TYPE_CHECKING or hasattr(typing, "GENERATING_DOCS"):
    from typing import Self

    SionnaScene: type | Any = Any

    try:
        import sionna.rt
    except ImportError:
        SionnaScene = Any
    else:
        SionnaScene = sionna.rt.Scene
else:
    Self = Any  # Because runtime type checking from 'beartype' will fail when combined with 'jaxtyping'
    SionnaScene = Any


from differt.geometry._mesh import (
    _WARP_MESHES_CACHE,  # TODO: should we create a separate cache here?
)


@no_type_check
@wp.func
def combine_hashes(h1: wp.uint32, h2: wp.uint32) -> wp.uint32:  # pragma: no cover
    return h1 ^ (
        h2 + wp.uint32(0x9E3779B9) + (h1 << wp.uint32(6)) + (h1 >> wp.uint32(2))
    )


@no_type_check
@wp.func
def hash_int(x: wp.uint32) -> wp.uint32:  # pragma: no cover
    x = ((x >> wp.uint32(16)) ^ x) * wp.uint32(0x45D9F3B)
    x = ((x >> wp.uint32(16)) ^ x) * wp.uint32(0x45D9F3B)
    return (x >> wp.uint32(16)) ^ x


@no_type_check
@wp.kernel
def _compute_tx_mlm_kernel(
    mesh_id: wp.uint64,
    mesh_points: wp.array[wp.vec3],
    mesh_indices: wp.array[wp.int32],
    ray_origins: wp.array(dtype=wp.vec3, ndim=2),
    ray_directions: wp.array(dtype=wp.vec3, ndim=2),
    dim_x: int,
    dim_y: int,
    max_order: int,
    min_order: int,
    assume_quads: bool,
    receiver_height: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    output: wp.array3d[wp.uint32],
) -> None:  # pragma: no cover
    itx, iray = wp.tid()

    current_origin = ray_origins[itx, iray]
    current_direction = ray_directions[itx, iray]
    ray_hash = wp.uint32(2166136261)

    epsilon = wp.float32(1e-4)
    dx = (wp.float32(max_x) - wp.float32(min_x)) / wp.float32(dim_x)
    dy = (wp.float32(max_y) - wp.float32(min_y)) / wp.float32(dim_y)

    for t in range(max_order + 1):
        # Query closest hit along the ray
        query_origin = current_origin
        if t > 0:
            query_origin = current_origin + current_direction * epsilon

        res = wp.mesh_query_ray(mesh_id, query_origin, current_direction, wp.inf)

        # Distance to closest triangle hit (if any)
        t_hit = wp.inf
        if res.result:
            t_hit = res.t + epsilon if t > 0 else res.t

        # Intersection with the receiver plane z = receiver_height
        if wp.abs(current_direction[2]) > wp.float32(1e-6):
            u = (wp.float32(receiver_height) - query_origin[2]) / current_direction[2]

            # Intersection point P
            P = query_origin + current_direction * u  # ruff:ignore[non-lowercase-variable-in-function]

            # Check if intersection is valid and unobstructed
            if u > wp.float32(0.0) and u < t_hit:  # ruff:ignore[collapsible-if]
                if t >= min_order and (
                    P[0] >= wp.float32(min_x)
                    and P[0] <= wp.float32(max_x)
                    and P[1] >= wp.float32(min_y)
                    and P[1] <= wp.float32(max_y)
                ):
                    # It hit the receiver grid!
                    ix = wp.int32(wp.floor((P[0] - wp.float32(min_x)) / dx))
                    iy = wp.int32(wp.floor((P[1] - wp.float32(min_y)) / dy))

                    # Clip to bounds
                    ix = wp.clamp(ix, wp.int32(0), wp.int32(dim_x - 1))
                    iy = wp.clamp(iy, wp.int32(0), wp.int32(dim_y - 1))

                    # Add path hash to cell
                    wp.atomic_or(output, itx, ix, iy, ray_hash)

        # If the ray hit a triangle, we bounce it
        if res.result:
            # Update origin to hit point
            current_origin = query_origin + current_direction * res.t

            # TODO: maybe we should pre-calculate the face normals?
            # Compute face normal
            face_index = res.face
            i0 = mesh_indices[face_index * 3 + 0]
            i1 = mesh_indices[face_index * 3 + 1]
            i2 = mesh_indices[face_index * 3 + 2]
            v0 = mesh_points[i0]
            v1 = mesh_points[i1]
            v2 = mesh_points[i2]

            # Normal vector
            normal = wp.normalize(wp.cross(v1 - v0, v2 - v0))

            # Reflected direction
            current_direction = (
                current_direction
                - wp.float32(2.0) * wp.dot(current_direction, normal) * normal
            )

            # Update path hash
            hash_face = face_index
            if assume_quads:
                hash_face = face_index // 2
            ray_hash = combine_hashes(ray_hash, hash_int(wp.uint32(hash_face)))
        else:
            # No hit, ray goes to infinity
            break


@no_type_check
def _compute_tx_mlm_func(
    mesh_id: int,
    mesh_points: wp.array[wp.vec3],
    mesh_indices: wp.array[wp.int32],
    ray_origins: wp.array(dtype=wp.vec3, ndim=2),
    ray_directions: wp.array(dtype=wp.vec3, ndim=2),
    dim_x: int,
    dim_y: int,
    num_rays: int,
    max_order: int,
    min_order: int,
    assume_quads: bool,
    receiver_height: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    output: wp.array(dtype=wp.uint32, ndim=3),
) -> None:
    if (wp_mesh := _WARP_MESHES_CACHE.get(mesh_id)) is None:
        wp_mesh = wp.Mesh(points=mesh_points, indices=mesh_indices)
        _WARP_MESHES_CACHE[mesh_id] = wp_mesh

    output.fill_(0)

    num_tx = ray_origins.shape[0]

    wp.launch(
        _compute_tx_mlm_kernel,
        dim=(num_tx, num_rays),
        inputs=[
            wp_mesh.id,
            mesh_points,
            mesh_indices,
            ray_origins,
            ray_directions,
            dim_x,
            dim_y,
            max_order,
            min_order,
            assume_quads,
            receiver_height,
            min_x,
            max_x,
            min_y,
            max_y,
        ],
        outputs=[output],
        device=ray_origins.device,
    )


@no_type_check
def _compute_tx_mlm_cuda_impl(
    mesh_id: np.uint64,
    vertices: Float[Array, "num_vertices 3"],
    triangles: Int[Array, "num_triangles 3"],
    ray_origins: Float[Array, "num_tx num_rays 3"],
    ray_directions: Float[Array, "num_tx num_rays 3"],
    *,
    dim_x: int,
    dim_y: int,
    num_rays: int,
    max_order: int,
    min_order: int,
    assume_quads: bool,
    receiver_height: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> Uint[Array, "num_tx dim_x dim_y"]:
    num_tx = ray_origins.shape[0]
    return wp.jax_callable(
        _compute_tx_mlm_func,
        num_outputs=1,
        output_dims=(num_tx, dim_x, dim_y),
        graph_mode=wp.JaxCallableGraphMode.NONE,
    )(
        mesh_id,
        vertices,
        triangles.ravel(),
        ray_origins,
        ray_directions,
        dim_x,
        dim_y,
        num_rays,
        max_order,
        min_order,
        assume_quads,
        receiver_height,
        min_x,
        max_x,
        min_y,
        max_y,
    )[0]


@no_type_check
def _compute_tx_mlm_cpu_impl(
    mesh_id: np.uint64,
    vertices: Float[Array, "num_vertices 3"],
    triangles: Int[Array, "num_triangles 3"],
    ray_origins: Float[Array, "num_tx num_rays 3"],
    ray_directions: Float[Array, "num_tx num_rays 3"],
    *,
    dim_x: int,
    dim_y: int,
    num_rays: int,
    max_order: int,
    min_order: int,
    assume_quads: bool,
    receiver_height: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> Uint[Array, "num_tx dim_x dim_y"]:
    num_tx = ray_origins.shape[0]

    def callback(
        jax_vertices: Float[Array, "num_vertices 3"],
        jax_triangles: Int[Array, "num_triangles 3"],
        jax_ray_origins: Float[Array, "num_tx num_rays 3"],
        jax_ray_directions: Float[Array, "num_tx num_rays 3"],
    ) -> Uint[Array, "num_tx dim_x dim_y"]:
        wp_vertices = wp.from_jax(jax_vertices, dtype=wp.vec3)
        wp_triangles = wp.from_jax(jax_triangles.ravel(), dtype=wp.int32)
        wp_ray_origins = wp.from_jax(jax_ray_origins, dtype=wp.vec3)
        wp_ray_directions = wp.from_jax(jax_ray_directions, dtype=wp.vec3)

        output = wp.empty(
            (num_tx, dim_x, dim_y), dtype=wp.uint32, device=wp_ray_origins.device
        )

        _compute_tx_mlm_func(
            int(mesh_id),
            wp_vertices,
            wp_triangles,
            wp_ray_origins,
            wp_ray_directions,
            dim_x,
            dim_y,
            num_rays,
            max_order,
            min_order,
            assume_quads,
            receiver_height,
            min_x,
            max_x,
            min_y,
            max_y,
            output,
        )

        return wp.to_jax(output)

    return jax.pure_callback(
        callback,
        jax.ShapeDtypeStruct((num_tx, dim_x, dim_y), jnp.uint32),
        vertices,
        triangles,
        ray_origins,
        ray_directions,
    )


@eqx.filter_jit
@no_type_check
def _compute_tx_mlm(
    tx: Float[Array, "num_tx 3"],
    mesh: Mesh,
    max_order: int,
    min_order: int,
    assume_quads: bool,
    dim_x: int,
    dim_y: int,
    num_rays: int,
    receiver_height: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> Uint[Array, "num_tx dim_x dim_y"]:
    # Prepare arrays
    points = mesh.vertices
    indices = mesh.triangles

    world_vertices = mesh.triangle_vertices.reshape(-1, 3)

    if mesh.mask is not None:
        active_vertices = jnp.repeat(mesh.mask, 3, axis=0)
        indices = jnp.where(mesh.mask[:, None], indices, 0)
    else:
        active_vertices = None

    # Include the 4 corner points of the receiver plane to expand viewing frustum
    corners = jnp.array([
        [min_x, min_y, receiver_height],
        [max_x, min_y, receiver_height],
        [max_x, max_y, receiver_height],
        [min_x, max_y, receiver_height],
    ])
    world_vertices = jnp.concatenate((world_vertices, corners), axis=0)
    if active_vertices is not None:
        active_vertices = jnp.concatenate(
            (active_vertices, jnp.ones(4, dtype=bool)), axis=0
        )

    def gen_rays(
        t: Float[Array, "3"],
    ) -> tuple[Float[Array, "num_rays 3"], Float[Array, "num_rays 3"]]:
        f = viewing_frustum(t, world_vertices, active_vertices=active_vertices)
        f = f.at[1, 1].set(jnp.pi)  # TODO: fixme
        origins = jnp.repeat(t[None, :], num_rays, axis=0)
        directions = fibonacci_lattice(num_rays, frustum=f)
        return origins, directions

    ray_origins, ray_directions = jax.vmap(gen_rays)(tx)

    mesh_id = np.uint64(id(mesh))

    return jax.lax.platform_dependent(
        points,
        indices,
        ray_origins,
        ray_directions,
        cpu=partial(
            _compute_tx_mlm_cpu_impl,
            mesh_id,
            dim_x=dim_x,
            dim_y=dim_y,
            num_rays=num_rays,
            max_order=max_order,
            min_order=min_order,
            assume_quads=assume_quads,
            receiver_height=receiver_height,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
        ),
        cuda=partial(
            _compute_tx_mlm_cuda_impl,
            mesh_id,
            dim_x=dim_x,
            dim_y=dim_y,
            num_rays=num_rays,
            max_order=max_order,
            min_order=min_order,
            assume_quads=assume_quads,
            receiver_height=receiver_height,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
        ),
    )


class Scene(eqx.Module):
    """A simple scene made of one or more triangle meshes, some transmitters and some receivers."""

    transmitters: Float[Array, "*transmitters_batch 3"] = eqx.field(
        default_factory=lambda: jnp.empty((0, 3)),
    )
    """The array of transmitter vertices."""
    receivers: Float[Array, "*receivers_batch 3"] = eqx.field(
        default_factory=lambda: jnp.empty((0, 3)),
    )
    """The array of receiver vertices."""
    mesh: Mesh = eqx.field(default_factory=Mesh.empty)
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
        Return a new instance of this scene with :attr:`Mesh.assume_quads<differt.geometry.Mesh.assume_quads>` set to ``flag``.

        This is simply a convenient wrapper to call :meth:`Mesh.set_assume_quads<differt.geometry.Mesh.set_assume_quads>` on the inner :attr:`mesh` attribute.

        Args:
            flag: The new flag value.

        Returns:
            A new scene with the same structure with the inner mesh's :attr:`Mesh.assume_quads<differt.geometry.Mesh.assume_quads>` set to ``flag``.
        """
        return eqx.tree_at(lambda s: s.mesh, self, self.mesh.set_assume_quads(flag))

    def with_transmitters_grid(
        self, m: int = 50, n: int | None = 50, *, height: Float[ArrayLike, ""] = 1.5
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
        self, m: int = 50, n: int | None = 50, *, height: Float[ArrayLike, ""] = 1.5
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
    def scale(self, scale_factor: Float[ArrayLike, ""]) -> Self:
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
    def from_core(cls, core_scene: differt_core.geometry.Scene) -> Self:
        """
        Return a triangle scene from a scene created by the :mod:`differt_core` module.

        Args:
            core_scene: The scene from the core module.

        Returns:
            The corresponding scene.
        """
        return cls(
            mesh=Mesh.from_core(core_scene.mesh),
        )

    @classmethod
    def load_xml(cls, file: str) -> Self:
        """
        Load a triangle scene from a XML file.

        This method uses
        :meth:`SionnaScene.load_xml<differt_core.geometry.SionnaScene.load_xml>`
        internally.

        Args:
            file: The path to the XML file.

        Returns:
            The corresponding scene containing only triangle meshes.
        """
        core_scene = differt_core.geometry.Scene.load_xml(file)
        return cls.from_core(core_scene)

    @classmethod
    def from_mitsuba(cls, mi_scene) -> Self:  # ruff:ignore[missing-type-function-argument]  # for some reason, mi.Scene cannot be imported, but only supports delayed annotations, which is not compatible with jaxtyping
        """
        Load a triangle scene from a Mitsuba scene object.

        This method does not extract any transmitters or receivers from the Mitsuba scene,
        as Mitsuba does not provide any explicit information about them, and they are usually
        part of the Sionna scene object, see :meth:`from_sionna`.

        Args:
            mi_scene (mitsuba.Scene): The Mitsuba scene object.

                You can obtain the Mitsuba scene object from a Sionna scene
                via its ``.mi_scene`` attribute.

        Returns:
            The corresponding scene containing only triangle meshes.

        .. seealso::

            :meth:`from_sionna`
        """
        mesh = Mesh.empty()

        for shape in mi_scene.shapes():
            rm = shape.bsdf().radio_material
            mesh += (
                Mesh(
                    vertices=shape.vertex_positions_buffer().jax().reshape(-1, 3),
                    triangles=shape.faces_buffer().jax().astype(int).reshape(-1, 3),
                )
                .set_face_colors(jnp.asarray(rm.color))
                .set_materials(f"itu_{rm.itu_type}")
                .set_face_materials(0)
            )

        return cls(
            mesh=mesh,
        )

    @classmethod
    def from_sionna(cls, sionna_scene: SionnaScene) -> Self:
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
    def trace_paths(
        self,
        order: None = ...,
        *,
        solver: Literal["exhaustive"] = "exhaustive",
        path_candidates: Int[ArrayLike, "num_path_candidates order"],
        **solver_kwargs: Unpack[_ExhaustivePathTracerKwargs],
    ) -> TracedPaths: ...

    @overload
    def trace_paths(
        self,
        order: None = ...,
        *,
        solver: Literal["hybrid"],
        path_candidates: Int[ArrayLike, "num_path_candidates order"],
        **solver_kwargs: Unpack[_HybridPathTracerKwargs],
    ) -> TracedPaths: ...

    @overload
    def trace_paths(
        self,
        order: None = ...,
        *,
        solver: AbstractPathTracer,
        path_candidates: Int[ArrayLike, "num_path_candidates order"],
    ) -> TracedPaths: ...

    @overload
    def trace_paths(
        self,
        order: int,
        *,
        solver: Literal["exhaustive"] = "exhaustive",
        path_candidates: None = ...,
        **solver_kwargs: Unpack[_ExhaustivePathTracerKwargs],
    ) -> TracedPaths | Iterator[TracedPaths]: ...

    @overload
    def trace_paths(
        self,
        order: int,
        *,
        solver: Literal["hybrid"],
        path_candidates: None = ...,
        **solver_kwargs: Unpack[_HybridPathTracerKwargs],
    ) -> TracedPaths | Iterator[TracedPaths]: ...

    @overload
    def trace_paths(
        self,
        order: int,
        *,
        solver: AbstractPathTracer,
        path_candidates: None = ...,
    ) -> TracedPaths | Iterator[TracedPaths]: ...

    def trace_paths(
        self,
        order: int | None = None,
        *,
        solver: AbstractPathTracer | Literal["exhaustive", "hybrid"] = "exhaustive",
        path_candidates: Int[ArrayLike, "num_path_candidates order"] | None = None,
        **solver_kwargs: Any,
    ) -> TracedPaths | SizedIterator[TracedPaths] | Iterator[TracedPaths]:
        """
        Trace paths between all pairs of transmitters and receivers in the scene, using exact methods (image method + validation).

        .. warning::

            This method is Warp-accelerated (via :class:`Mesh<differt.geometry.Mesh>`) and only supports CPU and CUDA-enabled GPU platforms.
            It does not support TPUs or other non-CUDA GPUs.

        Note:
            Currently, only :abbr:`LOS (line of sight)` and fixed ``order`` reflection paths are computed,
            using the :func:`image_method<differt.geometry.image_method>`. More types of interactions
            and path tracing methods will be added in the future, so stay tuned!

        Args:
            order: The number of interactions (bounces).
                This or ``path_candidates`` must be specified.
            solver: The solver configuration or string shortcut.
            path_candidates: An optional array of path candidates, see :ref:`path_candidates`.
                This is helpful to only generate paths on a subset of the scene.
                If :attr:`self.mesh.assume_quads<differt.geometry.Mesh.assume_quads>`
                is :data:`True`, then path candidates are rounded down toward the nearest
                even value.
            **solver_kwargs: Parameters passed  to the solver configuration when it is
                instantiated from a string shortcut. Any parameters that were also passed as
                arguments to the function call will override the corresponding values
                in the solver configuration.

        Returns:
            The traced paths.

        Raises:
            ValueError: If neither or both of ``order`` and ``path_candidates`` are
                specified, or if the solver shortcut is unknown.
        """
        if (order is None) == (path_candidates is None):
            msg = "You must specify one of 'order' or `path_candidates`, not both."
            raise ValueError(msg)

        if isinstance(solver, str):
            if solver == "exhaustive":
                solver = ExhaustivePathTracer(**solver_kwargs)
            elif solver == "hybrid":
                solver = HybridPathTracer(**solver_kwargs)
            else:
                msg = f"Unknown solver: {solver}"
                raise ValueError(msg)
        elif solver_kwargs:
            msg = "solver_kwargs cannot be used when a solver instance is provided."
            raise ValueError(msg)

        if (
            isinstance(solver, HybridPathTracer)
            and getattr(solver, "smoothing_factor", None) is not None
        ):
            warnings.warn(
                "Argument 'smoothing' is currently ignored when using HybridPathTracer.",
                UserWarning,
                stacklevel=2,
            )
        if isinstance(solver, HybridPathTracer) and order is None:
            msg = "Argument 'order' is required when using HybridPathTracer."
            raise ValueError(msg)
        if (path_candidates is not None) and getattr(
            solver, "chunk_size", None
        ) is not None:
            warnings.warn(
                "Argument 'chunk_size' is ignored when 'path_candidates' is provided.",
                UserWarning,
                stacklevel=2,
            )
            solver = dataclasses.replace(solver, chunk_size=None)
        tx_batch = self.transmitters.shape[:-1]
        rx_batch = self.receivers.shape[:-1]

        # TODO: simplify logic below:
        # if path_candidates is passed, then we can bypass the pass candidates generation (chunk_size is then ignored)

        if path_candidates is None:
            order = cast("int", order)
            chunk_size = getattr(solver, "chunk_size", None)
            if chunk_size is not None:
                chunks_iter = solver.generate_path_candidates_chunks_iter(
                    self, order, chunk_size=chunk_size
                )
                it: Iterator[TracedPaths] = (
                    solver.trace_path_candidates(
                        self, chunk_cands, chunk_types
                    ).reshape(*tx_batch, *rx_batch, chunk_cands.shape[0])
                    for chunk_cands, chunk_types in chunks_iter
                )
                if hasattr(chunks_iter, "__len__"):
                    size = cast("Callable[[], int]", chunks_iter.__len__)
                    return SizedIterator(it, size=size)
                return it

            candidates, interaction_types = solver.generate_path_candidates(self, order)
        else:
            path_candidates_arr = jnp.asarray(path_candidates)
            if self.mesh.assume_quads:
                path_candidates_arr -= path_candidates_arr % 2
            candidates = path_candidates_arr
            # Default: all specular reflections (value 0)
            interaction_types = jnp.zeros_like(candidates, dtype=jnp.int32)

        return solver.trace_path_candidates(
            self, candidates, interaction_types
        ).reshape(*tx_batch, *rx_batch, candidates.shape[0])

    @overload
    def launch_paths(
        self,
        order: int | None = ...,
        *,
        solver: Literal["sbr"] = "sbr",
        **solver_kwargs: Unpack[_SBRPathLauncherKwargs],
    ) -> LaunchedPaths: ...

    @overload
    def launch_paths(
        self,
        order: int | None = ...,
        *,
        solver: AbstractPathLauncher,
    ) -> LaunchedPaths: ...

    def launch_paths(
        self,
        order: int | None = None,
        *,
        solver: AbstractPathLauncher | Literal["sbr"] = "sbr",
        **solver_kwargs: Any,
    ) -> LaunchedPaths:
        """
        Launch paths from transmitters and find which paths are intercepted by receivers.

        .. warning::

            This method is Warp-accelerated (via :class:`Mesh<differt.geometry.Mesh>`) and only supports CPU and CUDA-enabled GPU platforms.
            It does not support TPUs or other non-CUDA GPUs.

        .. important::

            This SBR method is currently unstable and not yet optimized, and it is likely
            to change in future releases. Use with caution.

        Args:
            order: The maximum path order (number of interactions/bounces).
            solver: The solver configuration or string shortcut.
            **solver_kwargs: Keyword arguments passed to the solver when it is
                instantiated from a string shortcut.

        Returns:
            The launched paths.

        Raises:
            ValueError: If ``order`` is missing or the solver shortcut is unknown.
        """
        if order is None:
            msg = "Argument 'order' is required."
            raise ValueError(msg)

        if isinstance(solver, str):
            if solver == "sbr":
                solver = SBRPathLauncher(**solver_kwargs)
            else:
                msg = f"Unknown solver: {solver}"
                raise ValueError(msg)
        elif solver_kwargs:
            msg = "solver_kwargs cannot be used when a solver instance is provided."
            raise ValueError(msg)

        tx_batch = self.transmitters.shape[:-1]
        rx_batch = self.receivers.shape[:-1]

        return solver.launch_paths(
            self,
            order=order,
        ).reshape(*tx_batch, *rx_batch, -1)

    @overload
    def compute_paths(
        self,
        order: int | None = ...,
        *,
        method: Literal["exhaustive"] = ...,
        chunk_size: None = ...,
        num_rays: int = ...,
        path_candidates: Int[ArrayLike, "num_path_candidates order"] | None = ...,
        epsilon: Float[ArrayLike, ""] | None = ...,
        hit_tol: Float[ArrayLike, ""] | None = ...,
        min_len: Float[ArrayLike, ""] | None = ...,
        max_dist: Float[ArrayLike, ""] = ...,
        smoothing_factor: Float[ArrayLike, ""] | None = ...,
        confidence_threshold: Float[ArrayLike, ""] = ...,
        batch_size: int | None = ...,
        disconnect_inactive_triangles: bool = ...,
    ) -> TracedPaths: ...

    @overload
    def compute_paths(
        self,
        order: int | None = ...,
        *,
        method: Literal["exhaustive"] = ...,
        chunk_size: None = None,
        num_rays: int = ...,
        path_candidates: Int[ArrayLike, "num_path_candidates order"] | None = ...,
        epsilon: Float[ArrayLike, " "] | None = ...,
        hit_tol: Float[ArrayLike, " "] | None = ...,
        min_len: Float[ArrayLike, " "] | None = ...,
        max_dist: Float[ArrayLike, " "] = ...,
        smoothing_factor: None = ...,
        confidence_threshold: Float[ArrayLike, " "] = ...,
        batch_size: int | None = ...,
        disconnect_inactive_triangles: bool = ...,
    ) -> TracedPaths: ...

    @overload
    def compute_paths(
        self,
        order: int,
        *,
        method: Literal["hybrid"],
        chunk_size: None = ...,
        num_rays: int = ...,
        path_candidates: None = ...,
        epsilon: Float[ArrayLike, ""] | None = ...,
        hit_tol: Float[ArrayLike, ""] | None = ...,
        min_len: Float[ArrayLike, ""] | None = ...,
        max_dist: Float[ArrayLike, ""] = ...,
        smoothing_factor: Float[ArrayLike, ""] | None = ...,
        confidence_threshold: Float[ArrayLike, ""] = ...,
        batch_size: int | None = ...,
        disconnect_inactive_triangles: bool = ...,
    ) -> TracedPaths: ...

    @overload
    def compute_paths(
        self,
        order: int,
        *,
        method: Literal["hybrid"],
        chunk_size: None = None,
        num_rays: int = ...,
        path_candidates: None = ...,
        epsilon: Float[ArrayLike, " "] | None = ...,
        hit_tol: Float[ArrayLike, " "] | None = ...,
        min_len: Float[ArrayLike, " "] | None = ...,
        max_dist: Float[ArrayLike, " "] = ...,
        smoothing_factor: None = ...,
        confidence_threshold: Float[ArrayLike, " "] = ...,
        batch_size: int | None = ...,
        disconnect_inactive_triangles: bool = ...,
    ) -> TracedPaths: ...

    @overload
    def compute_paths(
        self,
        order: int | None = None,
        *,
        method: Literal["exhaustive"] = ...,
        chunk_size: int,
        num_rays: int = ...,
        path_candidates: None = ...,
        epsilon: Float[ArrayLike, ""] | None = ...,
        hit_tol: Float[ArrayLike, ""] | None = ...,
        min_len: Float[ArrayLike, ""] | None = ...,
        max_dist: Float[ArrayLike, ""] = ...,
        smoothing_factor: Float[ArrayLike, ""] | None = ...,
        confidence_threshold: Float[ArrayLike, ""] = ...,
        batch_size: int | None = ...,
        disconnect_inactive_triangles: bool = ...,
    ) -> SizedIterator[TracedPaths]: ...

    @overload
    def compute_paths(
        self,
        order: int | None = ...,
        *,
        method: Literal["exhaustive"] = ...,
        chunk_size: int,
        num_rays: int = ...,
        path_candidates: None = ...,
        epsilon: Float[ArrayLike, " "] | None = ...,
        hit_tol: Float[ArrayLike, " "] | None = ...,
        min_len: Float[ArrayLike, " "] | None = ...,
        max_dist: Float[ArrayLike, " "] = ...,
        smoothing_factor: None = ...,
        confidence_threshold: Float[ArrayLike, " "] = ...,
        batch_size: int | None = ...,
        disconnect_inactive_triangles: bool = ...,
    ) -> SizedIterator[TracedPaths]: ...

    @overload
    def compute_paths(
        self,
        order: int,
        *,
        method: Literal["hybrid"],
        chunk_size: int,
        num_rays: int = ...,
        path_candidates: None = ...,
        epsilon: Float[ArrayLike, ""] | None = ...,
        hit_tol: Float[ArrayLike, ""] | None = ...,
        min_len: Float[ArrayLike, ""] | None = ...,
        max_dist: Float[ArrayLike, ""] = ...,
        smoothing_factor: Float[ArrayLike, ""] | None = ...,
        confidence_threshold: Float[ArrayLike, ""] = ...,
        batch_size: int | None = ...,
        disconnect_inactive_triangles: bool = ...,
    ) -> Iterator[TracedPaths]: ...

    @overload
    def compute_paths(
        self,
        order: int,
        *,
        method: Literal["hybrid"],
        chunk_size: int,
        num_rays: int = ...,
        path_candidates: None = ...,
        epsilon: Float[ArrayLike, " "] | None = ...,
        hit_tol: Float[ArrayLike, " "] | None = ...,
        min_len: Float[ArrayLike, " "] | None = ...,
        max_dist: Float[ArrayLike, " "] = ...,
        smoothing_factor: None = ...,
        confidence_threshold: Float[ArrayLike, " "] = ...,
        batch_size: int | None = ...,
        disconnect_inactive_triangles: bool = ...,
    ) -> Iterator[TracedPaths]: ...

    @overload
    def compute_paths(
        self,
        order: int | None = ...,
        *,
        method: Literal["exhaustive"] = ...,
        chunk_size: int,
        num_rays: int = ...,
        path_candidates: Int[ArrayLike, "num_path_candidates order"],
        epsilon: Float[ArrayLike, ""] | None = ...,
        hit_tol: Float[ArrayLike, ""] | None = ...,
        min_len: Float[ArrayLike, ""] | None = ...,
        max_dist: Float[ArrayLike, ""] = ...,
        smoothing_factor: Float[ArrayLike, ""] | None = ...,
        confidence_threshold: Float[ArrayLike, ""] = ...,
        batch_size: int | None = ...,
        disconnect_inactive_triangles: bool = ...,
    ) -> TracedPaths: ...

    @overload
    def compute_paths(
        self,
        order: int | None = ...,
        *,
        method: Literal["exhaustive"] = ...,
        chunk_size: int,
        num_rays: int = ...,
        path_candidates: Int[ArrayLike, "num_path_candidates order"],
        epsilon: Float[ArrayLike, " "] | None = ...,
        hit_tol: Float[ArrayLike, " "] | None = ...,
        min_len: Float[ArrayLike, " "] | None = ...,
        max_dist: Float[ArrayLike, " "] = ...,
        smoothing_factor: None = ...,
        confidence_threshold: Float[ArrayLike, " "] = ...,
        batch_size: int | None = ...,
        disconnect_inactive_triangles: bool = ...,
    ) -> TracedPaths: ...

    @overload
    def compute_paths(
        self,
        order: int,
        *,
        method: Literal["sbr"],
        chunk_size: None = None,
        num_rays: int = ...,
        path_candidates: None = ...,
        epsilon: Float[ArrayLike, ""] | None = ...,
        hit_tol: None = ...,
        min_len: None = ...,
        max_dist: Float[ArrayLike, ""] = ...,
        smoothing_factor: None = ...,
        confidence_threshold: Float[ArrayLike, ""] = ...,
        batch_size: int | None = ...,
        disconnect_inactive_triangles: bool = ...,
    ) -> LaunchedPaths: ...

    def compute_paths(
        self,
        order: int | None = None,
        *,
        method: Literal["exhaustive", "sbr", "hybrid"] = "exhaustive",
        chunk_size: int | None = None,
        num_rays: int = int(1e6),
        path_candidates: Int[ArrayLike, "num_path_candidates order"] | None = None,
        epsilon: Float[ArrayLike, ""] | None = None,
        hit_tol: Float[ArrayLike, ""] | None = None,
        min_len: Float[ArrayLike, ""] | None = None,
        max_dist: Float[ArrayLike, ""] = 1e-3,
        smoothing_factor: Float[ArrayLike, ""] | None = None,
        confidence_threshold: Float[ArrayLike, ""] = 0.5,
        batch_size: int | None = 512,
        disconnect_inactive_triangles: bool = False,
    ) -> (
        TracedPaths | SizedIterator[TracedPaths] | Iterator[TracedPaths] | LaunchedPaths
    ):
        """
        Compute paths between all pairs of transmitters and receivers in the scene, that undergo a fixed number of interaction with objects.

        .. deprecated:: 0.10
            Use :meth:`trace_paths` or :meth:`launch_paths` instead.

        .. warning::

            This method is Warp-accelerated (via :class:`Mesh<differt.geometry.Mesh>`) and only supports CPU and CUDA-enabled GPU platforms.
            It does not support TPUs or other non-CUDA GPUs.

        Note:
            Currently, only :abbr:`LOS (line of sight)` and fixed ``order`` reflection paths are computed,
            using the :func:`image_method<differt.geometry.image_method>`. More types of interactions
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

                If :attr:`self.mesh.assume_quads<differt.geometry.Mesh.assume_quads>`
                is :data:`True`, then path candidates are
                rounded down toward the nearest even value (but object indices still refer
                to triangle indices, not quadrilateral indices).

                **Not compatible with** ``method == 'sbr'`` and ``method == 'hybrid'``.
            epsilon: Tolerance for checking ray / objects intersection, see
                :func:`ray_intersect_triangle<differt.geometry.ray_intersect_triangle>`.
            hit_tol: Tolerance for checking blockage (i.e., obstruction), see
                :func:`ray_intersect_any_triangle<differt.geometry.ray_intersect_any_triangle>`.

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

                See :func:`ray_intersect_any_triangle<differt.geometry.ray_intersect_any_triangle>`,
                :func:`triangles_visible_from_vertex<differt.geometry.triangles_visible_from_vertex>`,
                and :func:`first_triangle_hit_by_ray<differt.geometry.first_triangle_hit_by_ray>`
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

                If ``method == 'sbr'`` or ``method == 'hybrid'``, and ``order`` is not provided.

        .. [#f1] Passing the squared length/distance is useful to avoid computing square root values, which is expensive.
        """
        warnings.warn(
            "compute_paths is deprecated. Use trace_paths() or launch_paths() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if method == "sbr":
            if order is None:
                msg = "Argument 'order' is required."
                raise ValueError(msg)
            solver = SBRPathLauncher(
                num_rays=num_rays,
                max_dist=max_dist,
            )
            return self.launch_paths(order=order, solver=solver)
        if method == "hybrid":
            solver = HybridPathTracer(
                num_rays=num_rays,
                epsilon=epsilon,
                hit_tol=hit_tol,
                min_len=min_len,
                smoothing_factor=smoothing_factor,
                confidence_threshold=confidence_threshold,
                batch_size=batch_size,
                chunk_size=chunk_size,
            )
            return self.trace_paths(  # type: ignore[ty:no-matching-overload]
                order=order, solver=solver, path_candidates=path_candidates
            )
        # exhaustive
        solver = ExhaustivePathTracer(
            epsilon=epsilon,
            hit_tol=hit_tol,
            min_len=min_len,
            smoothing_factor=smoothing_factor,
            confidence_threshold=confidence_threshold,
            batch_size=batch_size,
            disconnect_inactive_triangles=disconnect_inactive_triangles,
            chunk_size=chunk_size,
        )
        return self.trace_paths(  # type: ignore[ty:no-matching-overload]
            order=order, solver=solver, path_candidates=path_candidates
        )

    def compute_tx_mlm(
        self,
        max_order: int,
        dim_x: int,
        dim_y: int,
        num_rays: int = int(1e6),
        min_order: int = 0,
        height: float | None = None,
    ) -> Uint[Array, "*transmitters_batch dim_x dim_y"]:
        """
        Compute the Multipath Lifetime Map (MLM) from the transmitter(s) for a moving receiver on a 2D grid in the XY plane.

        This method implements the MLM algorithm described in the paper
        *Comparing Differentiable and Dynamic Ray Tracing: Introducing the
        Multipath Lifetime Map* :cite:`mlm-eucap2025`.

        Rather than performing exhaustive ray tracing for each grid receiver (which is
        computationally expensive and has a large memory footprint, as shown in the
        :ref:`multipath_lifetime_map` tutorial notebook), this function uses a
        **shooting and bouncing ray (SBR)** approach to efficiently sample paths from
        the transmitter and resolve which receiver cells they intersect.

        Warning:
            Because this function relies on a stochastic SBR approach, there is a
            trade-off between grid density and ray count. When increasing the resolution
            of the grid (i.e., ``dim_x`` and ``dim_y``), you **must** increase ``num_rays``
            correspondingly. Otherwise, some grid cells will not be sampled by any rays,
            leading to "unreached" cells and visible noise/holes in the map.

        Args:
            max_order: The maximum path order (number of bounces).
            dim_x: The number of grid cells along the X-axis.
            dim_y: The number of grid cells along the Y-axis.
            num_rays: The number of rays to launch from the transmitter.
            min_order: The minimum path order (number of bounces).
            height: The height (altitude) at which the MLM is computed. If None,
                defaults to the height of the first receiver, or 1.5 if no receivers.

        Returns:
            A 2D array representing the path hashes for each grid cell.

        Examples:
            The following example demonstrates how to compute and visualize a 3D MLM
            for a simple street canyon scene.

            .. plotly::

                >>> from differt.geometry import Scene, get_sionna_scene
                >>> from differt.plotting import draw_image
                >>> import equinox as eqx
                >>>
                >>> # Load the simple street canyon scene
                >>> scene_path = get_sionna_scene("simple_street_canyon")
                >>> scene = Scene.load_xml(scene_path)
                >>> scene = eqx.tree_at(
                ...     lambda s: s.transmitters, scene, jnp.array([0.0, 0.0, 32.0])
                ... )
                >>>
                >>> # Define grid limits and compute the MLM at height z=1.5
                >>> bbox = scene.mesh.bounding_box
                >>> x = jnp.linspace(bbox[0, 0], bbox[1, 0], 100)
                >>> y = jnp.linspace(bbox[0, 1], bbox[1, 1], 100)
                >>> mlm = scene.compute_tx_mlm(
                ...     max_order=2,
                ...     dim_x=100,
                ...     dim_y=100,
                ...     height=1.5,
                ... )
                >>>
                >>> # Map hashes to random colors, masking out the background (hash=0)
                >>> mlm = mlm.T  # Transpose to swap x and y axes
                >>> cell_colors = jnp.vectorize(
                ...     lambda h: jr.uniform(jr.key(h), shape=(4,)).at[3].set(1),
                ...     signature="()->(4)",
                ... )(mlm)
                >>> cell_colors = jnp.where(mlm[..., None] == 0, 0, cell_colors)
                >>>
                >>> # Plot scene and overlay the computed MLM at the target height
                >>> fig = scene.plot(backend="plotly")
                >>> fig = draw_image(
                ...     cell_colors,
                ...     x=x,
                ...     y=y,
                ...     z0=1.5,
                ...     figure=fig,
                ...     backend="plotly",
                ... )
                >>> fig  # doctest: +SKIP
        """
        tx_shape = self.transmitters.shape[:-1]
        tx_flat = jax.lax.stop_gradient(self.transmitters).reshape(-1, 3)
        if height is not None:
            receiver_height = height
        elif self.receivers.size > 0:
            receiver_height = float(self.receivers.reshape(-1, 3)[0, 2])
        else:
            receiver_height = 1.5

        bbox = self.mesh.bounding_box
        min_x = float(bbox[0, 0])
        max_x = float(bbox[1, 0])
        min_y = float(bbox[0, 1])
        max_y = float(bbox[1, 1])

        out = _compute_tx_mlm(
            tx_flat,
            self.mesh,
            max_order=max_order,
            min_order=min_order,
            assume_quads=self.mesh.assume_quads,
            dim_x=dim_x,
            dim_y=dim_y,
            num_rays=num_rays,
            receiver_height=receiver_height,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
        )

        res = out.reshape(*tx_shape, dim_x, dim_y)
        return jax.lax.stop_gradient(res)

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
                :meth:`Mesh.plot<differt.geometry.Mesh.plot>`.
            kwargs: Keyword arguments passed to
                :func:`reuse<differt.plotting.reuse>`.

        Returns:
            The resulting plot output.
        """
        tx_kwargs: dict[str, Any] = {"labels": "tx", **(tx_kwargs or {})}
        rx_kwargs: dict[str, Any] = {"labels": "rx", **(rx_kwargs or {})}
        mesh_kwargs: Mapping[str, Any] = {} if mesh_kwargs is None else mesh_kwargs

        with reuse(pass_all_kwargs=True, **kwargs) as result:
            if self.transmitters.size > 0:
                draw_markers(self.transmitters, **tx_kwargs)

            if self.receivers.size > 0:
                draw_markers(self.receivers, **rx_kwargs)

            self.mesh.plot(**mesh_kwargs)

        return result


# Deprecated alias
class TriangleScene(Scene):
    """
    Deprecated alias for :class:`Scene`.

    .. deprecated:: 0.10
        Use :class:`Scene` instead.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "TriangleScene is deprecated, use Scene instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
