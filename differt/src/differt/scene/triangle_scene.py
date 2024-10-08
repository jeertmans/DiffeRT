"""Scene made of triangles and utilities."""
# ruff: noqa: ERA001

import sys
import warnings
from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype as typechecker
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P  # noqa: N814
from jaxtyping import Array, ArrayLike, Bool, Float, Int, jaxtyped

import differt_core.scene.triangle_scene
from differt.geometry.paths import Paths
from differt.geometry.triangle_mesh import (
    TriangleMesh,
)
from differt.geometry.utils import assemble_paths
from differt.plotting import draw_markers, reuse
from differt.rt.image_method import (
    consecutive_vertices_are_on_same_side_of_mirrors,
    image_method,
)
from differt.rt.utils import (
    SizedIterator,
    generate_all_path_candidates,
    generate_all_path_candidates_chunks_iter,
    rays_intersect_any_triangle,
    rays_intersect_triangles,
)

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def _compute_paths(
    mesh: TriangleMesh,
    from_vertices: Float[Array, "num_from_vertices 3"],
    to_vertices: Float[Array, "num_to_vertices 3"],
    path_candidates: Int[Array, "num_path_candidates order"],
    *,
    parallel: bool = False,
    epsilon: Float[ArrayLike, " "] | None = None,
    hit_tol: Float[ArrayLike, " "] | None = None,
    min_len: Float[ArrayLike, " "] | None = None,
) -> Paths:
    if min_len is None:
        dtype = jnp.result_type(mesh.vertices, from_vertices, to_vertices)
        min_len = 10 * jnp.finfo(dtype).eps

    # 1 - Broadcast arrays

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

    @jaxtyped(typechecker=typechecker)
    def fun(
        from_vertices: Float[Array, "num_from_vertices 3"],
        to_vertices: Float[Array, "num_to_vertices 3"],
    ) -> tuple[
        Float[
            Array, "num_from_vertices num_to_vertices num_path_candidates path_length 3"
        ],
        Bool[Array, "num_from_vertices num_to_vertices num_path_candidates"],
    ]:
        # 2 - Trace paths

        # [num_from_vertices num_to_vertices num_path_candidates order 3]
        paths = image_method(
            from_vertices[:, None, None, :],
            to_vertices[None, :, None, :],
            mirror_vertices,
            mirror_normals,
        )

        # [num_from_vertices num_to_vertices num_path_candidates order+2 3]
        full_paths = assemble_paths(
            from_vertices[:, None, None, None, :],
            paths,
            to_vertices[None, :, None, None, :],
        )

        # 3 - Identify invalid paths

        # [num_from_vertices num_to_vertices num_path_candidates order+1 3]
        ray_origins = full_paths[..., :-1, :]
        # [num_from_vertices num_to_vertices num_path_candidates order+1 3]
        ray_directions = jnp.diff(full_paths, axis=-2)

        # 3.1 - Check if paths vertices are inside respective triangles

        # [num_from_vertices num_to_vertices num_path_candidates]
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

        # [num_from_vertices num_to_vertices num_path_candidates]
        valid_reflections = consecutive_vertices_are_on_same_side_of_mirrors(
            full_paths,
            mirror_vertices,
            mirror_normals,
        ).all(axis=-1)  # Reduce on 'order'

        # 3.3 - Identify paths that are blocked by other objects

        # [num_from_vertices num_to_vertices num_path_candidates]
        blocked = rays_intersect_any_triangle(
            ray_origins,
            ray_directions,
            mesh.triangle_vertices,
            epsilon=epsilon,
            hit_tol=hit_tol,
        ).any(axis=-1)  # Reduce on 'order'

        # 3.4 - Identify path segments that are too small (e.g., double-reflection inside an edge)

        ray_lengths = jnp.sum(ray_directions * ray_directions, axis=-1)  # Squared norm

        too_small = (ray_lengths < min_len).any(
            axis=-1
        )  # Any path segment being too small

        mask = inside_triangles & valid_reflections & ~blocked & ~too_small

        return full_paths, mask

    if parallel:
        num_devices = jax.device_count()

        # TODO: allow also to have i,i mesh if product of both is a multiple of 'num_devices'
        if from_vertices.shape[0] % num_devices == 0:
            in_specs = (P("i", None), P(None, None))
            out_specs = (P("i", None, None, None, None), P("i", None, None))
        elif to_vertices.shape[0] % num_devices == 0:
            in_specs = (P(None, None), P("i", None))
            out_specs = (P(None, "i", None, None, None), P(None, "i", None))
        else:
            msg = (
                f"Found {num_devices} devices available, "
                "but could not find any input with a size that is a multiple of that value. "
                "Please user a number of transmitter or receiver points that is a "
                f"multiple of {num_devices}."
            )
            raise ValueError(msg)

        fun = shard_map(  # type: ignore[reportAssigmentType]
            fun,
            Mesh(mesh_utils.create_device_mesh((num_devices,)), axis_names=("i",)),
            in_specs=in_specs,
            out_specs=out_specs,
        )

    vertices, mask = fun(from_vertices, to_vertices)

    # 4 - Generate output paths and reshape

    object_dtype = path_candidates.dtype

    tx_objects = jnp.arange(from_vertices.shape[0], dtype=object_dtype)
    rx_objects = jnp.arange(to_vertices.shape[0], dtype=object_dtype)

    tx_objects = jnp.broadcast_to(
        tx_objects[:, None, None, None],
        (from_vertices.shape[0], to_vertices.shape[0], num_path_candidates, 1),
    )
    rx_objects = jnp.broadcast_to(
        rx_objects[None, :, None, None],
        (from_vertices.shape[0], to_vertices.shape[0], num_path_candidates, 1),
    )
    path_candidates = jnp.broadcast_to(
        path_candidates,
        (
            from_vertices.shape[0],
            to_vertices.shape[0],
            num_path_candidates,
            path_candidates.shape[1],
        ),
    )

    objects = jnp.concatenate((tx_objects, path_candidates, rx_objects), axis=-1)

    return Paths(
        vertices,
        objects,
        mask,
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

    @eqx.filter_jit
    @jaxtyped(
        typechecker=None
    )  # typing.Self is (currently) not compatible with jaxtyping and beartype
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

    @eqx.filter_jit
    @jaxtyped(
        typechecker=None
    )  # typing.Self is (currently) not compatible with jaxtyping and beartype
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

    @classmethod
    def from_core(
        cls, core_scene: differt_core.scene.triangle_scene.TriangleScene
    ) -> Self:
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
        self,
        order: int | None = None,
        *,
        chunk_size: int | None = None,
        path_candidates: Int[Array, "num_path_candidates order"] | None = None,
        parallel: bool = False,
        epsilon: Float[ArrayLike, " "] | None = None,
        hit_tol: Float[ArrayLike, " "] | None = None,
        min_len: Float[ArrayLike, " "] | None = None,
    ) -> Paths | SizedIterator[Paths]:
        """
        Compute paths between all pairs of transmitters and receivers in the scene, that undergo a fixed number of interaction with objects.

        Note:
            Currently, only :abbr:`LOS (line of sight)` and fixed ``order`` reflection paths are computed,
            using the :func:`image_method<differt.rt.image_method.image_method>`. More types of interactions
            and path tracing methods will be added in the future, so stay tuned!

        Args:
            order: The number of interaction, i.e., the number of bounces.

                This or ``path_candidates`` must be specified.
            chunk_size: If specified, it will iterate through chunks of path
                candidates, and yield the result as an iterator over paths chunks.

                Unused if ``path_candidates`` is provided.
            path_candidates: An option array of path candidates, see :ref:`path_candidates`.

                This is helpful to only generate paths on a subset of the scene.

                If ``self.mesh.assume_quads`` is :data:`True`, then path candidates are
                rounded down toward the nearest even value.
            parallel: If :data:`True`, ray tracing is performed in parallel across all available
                devices. Either the number of transmitters or the number of receivers
                **must** be a multiple of :func:`jax.device_count`, otherwise an error is raised.
            epsilon: Tolelance for checking ray / objects intersection, see
                :func:`rays_intersect_triangles<differt.rt.utils.rays_intersect_triangles>`.
            hit_tol: Tolerance for checking blockage (i.e., obstruction), see
                :func:`rays_intersect_any_triangle<differt.rt.utils.rays_intersect_any_triangle>`.
            min_len: Minimal (squared) length that each path segment must have for a path to be valid.

                If not specified, the default is ten times the epsilon value
                of the currently used floating point dtype.

        Returns:
            The paths, as class wrapping path vertices, object indices, and a masked
            identify valid paths.

        Raises:
            ValueError: If neither ``order`` nor ``path_candidates`` has been provided,
                or if both have been provided simultaneously.
        """
        if (order is None) == (path_candidates is None):
            msg = "You must specify one of 'order' or `path_candidates`, not both."
            raise ValueError(msg)
        if (chunk_size is not None) and (path_candidates is not None):
            msg = "Argument 'chunk_size' is ignored when 'path_candidates' is provided."
            warnings.warn(msg, UserWarning, stacklevel=2)
            chunk_size = None

        # 0 - Constants arrays of chunks
        num_objects = (
            self.mesh.num_quads if self.mesh.assume_quads else self.mesh.num_triangles
        )
        tx_batch = self.transmitters.shape[:-1]
        rx_batch = self.receivers.shape[:-1]

        # [tx_batch_flattened 3]
        from_vertices = self.transmitters.reshape(-1, 3)
        # [rx_batch_flattened 3]
        to_vertices = self.receivers.reshape(-1, 3)

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
                    from_vertices,
                    to_vertices,
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
                order,  # type: ignore[reportArgumentType]
            )

            if self.mesh.assume_quads:
                path_candidates = 2 * path_candidates
        elif self.mesh.assume_quads:
            path_candidates -= path_candidates % 2

        return _compute_paths(
            self.mesh,
            from_vertices,
            to_vertices,
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
