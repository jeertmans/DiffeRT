import typing
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NotRequired,
    TypedDict,
    TypeVar,
    Unpack,
    no_type_check,
    overload,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from jaxtyping import Array, ArrayLike, Bool, Float, Int, PRNGKeyArray

import differt_core.geometry
from differt.plotting import PlotOutput, draw_mesh, draw_paths, draw_rays, reuse

from ._utils import (
    fibonacci_lattice,
    normalize,
    orthogonal_basis,
    rotation_matrix_along_axis,
    viewing_frustum,
)

if TYPE_CHECKING or hasattr(typing, "GENERATING_DOCS"):
    from typing import Self
else:
    Self = Any  # Because runtime type checking from 'beartype' will fail when combined with 'jaxtyping'


# TODO: still allow setting log level and initialization from environ variable?
wp.config.log_level = wp.LOG_ERROR
wp.init()

# NOTE: Cache meshes to avoid re-creating them over and over.
# A problem with the current implementation is that @eqx.filter_jit
# creates a new TriangleMesh instance every time the function is recompiled,
# which creates cache misses. We could create a 'permanent' id for each mesh,
# e.g., when instantiating the TriangleMesh instance and passing it around,
# only updating it when it changes. However, this also means that we must keep
# track of all TriangleMesh instances that point to the same id.
_WARP_MESHES_CACHE: dict[int, wp.Mesh] = {}


class _AtIndexingKwargs(TypedDict):
    mode: NotRequired[Literal["promise_in_bounds", "clip", "drop", "fill"]]
    wrap_negative_indices: NotRequired[bool]
    indices_are_sorted: NotRequired[bool]
    unique_indices: NotRequired[bool]


_AT_INDEXING_KWARGS: _AtIndexingKwargs = {
    "wrap_negative_indices": False,
    "indices_are_sorted": True,
    "unique_indices": True,
}


class _GetIndexingKwargs(TypedDict):
    mode: Literal["promise_in_bounds", "clip", "drop", "fill"]
    wrap_negative_indices: NotRequired[bool]
    fill_value: NotRequired[Any]
    indices_are_sorted: NotRequired[bool]
    unique_indices: NotRequired[bool]


@jax.jit
def triangle_contains_vertex_assuming_inside_same_plane(
    triangle_vertices: Float[ArrayLike, "*#batch 3 3"],
    vertex: Float[ArrayLike, "*#batch 3"],
) -> Bool[Array, " *#batch"]:
    """
    Return whether each triangle contains the corresponding vertex, but assuming the vertex lies in the same plane as the triangle.

    This is especially useful when combined with the
    :func:`image_method<differt.rt.image_method>`, as the paths returned
    will also lie in the same plane as the mirrors, but may be outside of the actual reflector,
    e.g., a triangular surface.

    Args:
        triangle_vertices: Triangle vertices.
        vertex: Vertex that will be checked.

    Returns:
        A boolean array indicating whether vertices are in the corresponding triangles or not.
    """
    triangle_vertices = jnp.asarray(triangle_vertices)
    vertex = jnp.asarray(vertex)

    # [*batch 3]
    p0 = triangle_vertices[..., 0, :]
    p1 = triangle_vertices[..., 1, :]
    p2 = triangle_vertices[..., 2, :]

    # Vectors from test vertex to every triangle vertex
    # [*batch 3]
    u0 = p0 - vertex
    u1 = p1 - vertex
    u2 = p2 - vertex

    # Vectors from one triangle vertex to the next
    # [*batch 3]
    v0 = p1 - p0
    v1 = p2 - p1
    v2 = p0 - p2

    # Cross product between corresponding vectors,
    # resulting 'normal' vector should all be perpendicular
    # to the triangle surface
    # [*batch 3]
    n0 = jnp.cross(u0, v0)
    n1 = jnp.cross(u1, v1)
    n2 = jnp.cross(u2, v2)

    # Dot product between all pairs of 'normal' vectors
    # [*batch]
    d01 = jnp.sum(n0 * n1, axis=-1)
    d12 = jnp.sum(n1 * n2, axis=-1)
    d20 = jnp.sum(n2 * n0, axis=-1)

    # [*batch]
    all_pos = (d01 >= 0.0) & (d12 >= 0.0) & (d20 >= 0.0)
    all_neg = (d01 <= 0.0) & (d12 <= 0.0) & (d20 <= 0.0)

    # The vertices are contained if all signs are the same
    return all_pos | all_neg


@no_type_check
@wp.kernel
def _ray_intersect_any_triangle_kernel(
    mesh_id: wp.uint64,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    max_t: wp.array[wp.float32],
    output: wp.array[wp.bool],
) -> None:  # pragma: no cover
    tid = wp.tid()
    output[tid] = wp.mesh_query_ray_anyhit(
        mesh_id,
        ray_origins[tid],
        ray_directions[tid],
        max_t[tid],
    )


@no_type_check
def _ray_intersect_any_triangle_anyhit_func(
    mesh_id: int,
    points: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    max_t: wp.array[wp.float32],
    output: wp.array[wp.bool],
) -> None:
    if (wp_mesh := _WARP_MESHES_CACHE.get(mesh_id)) is None:
        wp_mesh = wp.Mesh(points=points, indices=indices)
        _WARP_MESHES_CACHE[mesh_id] = wp.Mesh(points=points, indices=indices)
    wp.launch(
        _ray_intersect_any_triangle_kernel,
        dim=ray_origins.shape[0],
        inputs=[wp_mesh.id, ray_origins, ray_directions, max_t],
        outputs=[output],
        device=ray_origins.device,
    )


def _ray_intersect_any_triangle_cuda_impl(
    mesh_id: np.uint64,
    vertices: Float[Array, "num_vertices 3"],
    triangles: Int[Array, "num_triangles 3"],
    ray_origins: Float[Array, "num_rays 3"],
    ray_directions: Float[Array, "num_rays 3"],
    max_t: Float[Array, " num_rays"],
) -> Bool[Array, " num_rays"]:
    return wp.jax_callable(
        _ray_intersect_any_triangle_anyhit_func,
        output_dims=(ray_origins.shape[0],),
        graph_mode=wp.JaxCallableGraphMode.NONE,
    )(
        mesh_id,
        vertices,
        triangles.ravel(),
        ray_origins,
        ray_directions,
        max_t,
    )[0]


def _ray_intersect_any_triangle_cpu_impl(
    mesh_id: np.uint64,
    vertices: Float[Array, "num_vertices 3"],
    triangles: Int[Array, "num_triangles 3"],
    ray_origins: Float[Array, "num_rays 3"],
    ray_directions: Float[Array, "num_rays 3"],
    max_t: Float[Array, " num_rays"],
) -> Bool[Array, " num_rays"]:
    def callback(
        jax_vertices: Float[Array, "num_vertices 3"],
        jax_triangles: Int[Array, "num_triangles 3"],
        jax_ray_origins: Float[Array, "num_rays 3"],
        jax_ray_directions: Float[Array, "num_rays 3"],
        jax_max_t: Float[Array, " num_rays"],
    ) -> Bool[Array, " num_rays"]:
        wp_vertices = wp.from_jax(jax_vertices, dtype=wp.vec3)
        wp_triangles = wp.from_jax(jax_triangles.ravel(), dtype=wp.int32)
        wp_ray_origins = wp.from_jax(jax_ray_origins, dtype=wp.vec3)
        wp_ray_directions = wp.from_jax(jax_ray_directions, dtype=wp.vec3)
        wp_max_t = wp.from_jax(jax_max_t)

        output = wp.empty(
            jax_ray_origins.shape[0], dtype=bool, device=wp_ray_origins.device
        )

        _ray_intersect_any_triangle_anyhit_func(
            int(mesh_id),
            wp_vertices,
            wp_triangles,
            wp_ray_origins,
            wp_ray_directions,
            wp_max_t,
            output,
        )

        return wp.to_jax(output)

    return jax.pure_callback(
        callback,
        jax.ShapeDtypeStruct((ray_origins.shape[0],), bool),
        vertices,
        triangles,
        ray_origins,
        ray_directions,
        max_t,
    )


@wp.kernel
@no_type_check
def _first_triangle_hit_by_ray_kernel(
    mesh_id: wp.uint64,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    epsilon: float,
    output_face: wp.array[wp.int32],
    output_dist: wp.array[wp.float32],
) -> None:  # pragma: no cover
    tid = wp.tid()
    origin = ray_origins[tid] + ray_directions[tid] * epsilon
    res = wp.mesh_query_ray(mesh_id, origin, ray_directions[tid], wp.inf)
    hit = res.result
    output_face[tid] = res.face if hit else -1
    output_dist[tid] = (res.t + epsilon) if hit else wp.inf


@no_type_check
def _first_triangle_hit_by_ray_func(
    mesh_id: int,
    points: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    output_face: wp.array[wp.int32],
    output_dist: wp.array[wp.float32],
) -> None:
    if (wp_mesh := _WARP_MESHES_CACHE.get(mesh_id)) is None:
        wp_mesh = wp.Mesh(points=points, indices=indices)
        _WARP_MESHES_CACHE[mesh_id] = wp.Mesh(points=points, indices=indices)
    epsilon = 1e-5
    wp.launch(
        _first_triangle_hit_by_ray_kernel,
        dim=ray_origins.shape[0],
        inputs=[wp_mesh.id, ray_origins, ray_directions, epsilon],
        outputs=[output_face, output_dist],
        device=ray_origins.device,
    )


def _first_triangle_hit_by_ray_cuda_impl(
    mesh_id: np.uint64,
    vertices: Float[Array, "num_vertices 3"],
    triangles: Int[Array, "num_triangles 3"],
    ray_origins: Float[Array, "num_rays 3"],
    ray_directions: Float[Array, "num_rays 3"],
) -> tuple[Int[Array, " num_rays"], Float[Array, " num_rays"]]:
    return tuple(
        wp.jax_callable(
            _first_triangle_hit_by_ray_func,
            num_outputs=2,
            output_dims=(ray_origins.shape[0],),
            graph_mode=wp.JaxCallableGraphMode.NONE,
        )(
            mesh_id,
            vertices,
            triangles.ravel(),
            ray_origins,
            ray_directions,
        )
    )


def _first_triangle_hit_by_ray_cpu_impl(
    mesh_id: np.uint64,
    vertices: Float[Array, "num_vertices 3"],
    triangles: Int[Array, "num_triangles 3"],
    ray_origins: Float[Array, "num_rays 3"],
    ray_directions: Float[Array, "num_rays 3"],
) -> tuple[Int[Array, " num_rays"], Float[Array, " num_rays"]]:
    def callback(
        jax_vertices: Float[Array, "num_vertices 3"],
        jax_triangles: Int[Array, "num_triangles 3"],
        jax_ray_origins: Float[Array, "num_rays 3"],
        jax_ray_directions: Float[Array, "num_rays 3"],
    ) -> tuple[Int[Array, " num_rays"], Float[Array, " num_rays"]]:
        wp_vertices = wp.from_jax(jax_vertices, dtype=wp.vec3)
        wp_triangles = wp.from_jax(jax_triangles.ravel(), dtype=wp.int32)
        wp_ray_origins = wp.from_jax(jax_ray_origins, dtype=wp.vec3)
        wp_ray_directions = wp.from_jax(jax_ray_directions, dtype=wp.vec3)

        output_faces = wp.empty(
            jax_ray_origins.shape[0], dtype=int, device=wp_ray_origins.device
        )
        output_dists = wp.empty(
            jax_ray_origins.shape[0], dtype=float, device=wp_ray_origins.device
        )

        _first_triangle_hit_by_ray_func(
            int(mesh_id),
            wp_vertices,
            wp_triangles,
            wp_ray_origins,
            wp_ray_directions,
            output_faces,
            output_dists,
        )

        return wp.to_jax(output_faces), wp.to_jax(output_dists)

    return jax.pure_callback(
        callback,
        (
            jax.ShapeDtypeStruct((ray_origins.shape[0],), jnp.int32),
            jax.ShapeDtypeStruct((ray_origins.shape[0],), jnp.float32),
        ),
        vertices,
        triangles,
        ray_origins,
        ray_directions,
    )


def _differentiable_distance(
    vertices: Float[Array, "num_vertices 3"],
    ray_origins: Float[Array, "num_rays 3"],
    ray_directions: Float[Array, "num_rays 3"],
    out_faces: Int[Array, " num_rays"],
    triangles: Int[Array, "num_triangles 3"],
) -> Float[Array, " num_rays"]:
    triangle_vertices = vertices[triangles]
    hit_triangle_vertices = triangle_vertices[out_faces]

    v0 = hit_triangle_vertices[:, 0, :]
    v1 = hit_triangle_vertices[:, 1, :]
    v2 = hit_triangle_vertices[:, 2, :]

    edge1 = v1 - v0
    edge2 = v2 - v0

    h = jnp.cross(ray_directions, edge2)
    a = jnp.sum(h * edge1, axis=-1)

    a = jnp.where(a == 0.0, jnp.inf, a)
    f = 1.0 / a
    s = ray_origins - v0
    q = jnp.cross(s, edge1)

    t = f * jnp.sum(q * edge2, axis=-1)

    is_hit = out_faces != -1

    return jnp.where(is_hit, t, jnp.inf)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _first_triangle_hit_by_ray_helper(
    mesh_id: np.uint64,
    vertices: Float[Array, "num_vertices 3"],
    triangles: Int[Array, "num_triangles 3"],
    flat_ray_origins: Float[Array, "num_rays 3"],
    flat_ray_directions: Float[Array, "num_rays 3"],
) -> tuple[Int[Array, " num_rays"], Float[Array, " num_rays"]]:
    out_faces, out_t = jax.lax.platform_dependent(
        vertices,
        triangles,
        flat_ray_origins,
        flat_ray_directions,
        cpu=partial(_first_triangle_hit_by_ray_cpu_impl, mesh_id),
        cuda=partial(_first_triangle_hit_by_ray_cuda_impl, mesh_id),
    )
    return out_faces, out_t


def _first_triangle_hit_by_ray_helper_fwd(
    mesh_id: np.uint64,
    vertices: Float[Array, "num_vertices 3"],
    triangles: Int[Array, "num_triangles 3"],
    flat_ray_origins: Float[Array, "num_rays 3"],
    flat_ray_directions: Float[Array, "num_rays 3"],
) -> tuple[
    tuple[Int[Array, " num_rays"], Float[Array, " num_rays"]],
    tuple[
        Float[Array, "num_vertices 3"],
        Int[Array, "num_triangles 3"],
        Float[Array, "num_rays 3"],
        Float[Array, "num_rays 3"],
        Int[Array, " num_rays"],
    ],
]:
    out_faces, out_t = _first_triangle_hit_by_ray_helper(
        mesh_id, vertices, triangles, flat_ray_origins, flat_ray_directions
    )
    return (out_faces, out_t), (
        vertices,
        triangles,
        flat_ray_origins,
        flat_ray_directions,
        out_faces,
    )


def _first_triangle_hit_by_ray_helper_bwd(
    _mesh_id: np.uint64,
    res: tuple[
        Float[Array, "num_vertices 3"],
        Int[Array, "num_triangles 3"],
        Float[Array, "num_rays 3"],
        Float[Array, "num_rays 3"],
        Int[Array, " num_rays"],
    ],
    g: tuple[Any, Float[Array, " num_rays"]],
) -> tuple[
    Float[Array, "num_vertices 3"] | None,
    None,
    Float[Array, "num_rays 3"] | None,
    Float[Array, "num_rays 3"] | None,
]:
    vertices, triangles, flat_ray_origins, flat_ray_directions, out_faces = res
    _, grad_t = g

    def diff_fun(
        v: Float[Array, "num_vertices 3"],
        ro: Float[Array, "num_rays 3"],
        rd: Float[Array, "num_rays 3"],
    ) -> Float[Array, " num_rays"]:
        return _differentiable_distance(v, ro, rd, out_faces, triangles)

    _, vjp_fn = jax.vjp(diff_fun, vertices, flat_ray_origins, flat_ray_directions)

    grad_vertices, grad_ray_origins, grad_ray_directions = vjp_fn(grad_t)

    return grad_vertices, None, grad_ray_origins, grad_ray_directions


_first_triangle_hit_by_ray_helper.defvjp(
    _first_triangle_hit_by_ray_helper_fwd,
    _first_triangle_hit_by_ray_helper_bwd,
)


@no_type_check
@wp.kernel
def _triangles_visible_from_vertex_kernel(
    mesh_id: wp.uint64,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    epsilon: float,
    num_rays: int,
    num_triangles: int,
    output_visible: wp.array[wp.bool],
) -> None:  # pragma: no cover
    tid = wp.tid()
    batch_idx = tid // num_rays

    origin = ray_origins[tid] + ray_directions[tid] * epsilon
    res = wp.mesh_query_ray(mesh_id, origin, ray_directions[tid], wp.inf)

    if res.result:
        # Concurrent writes of 'True' to the same address are benign and safe here
        output_visible[batch_idx * num_triangles + res.face] = True


@no_type_check
def _triangles_visible_from_vertex_func(
    mesh_id: int,
    points: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    num_rays: int,
    num_triangles: int,
    output_visible: wp.array[wp.bool],
) -> None:
    if (wp_mesh := _WARP_MESHES_CACHE.get(mesh_id)) is None:
        wp_mesh = wp.Mesh(points=points, indices=indices)
        _WARP_MESHES_CACHE[mesh_id] = wp.Mesh(points=points, indices=indices)

    epsilon = 1e-5
    output_visible.fill_(False)  # noqa: FBT003

    wp.launch(
        _triangles_visible_from_vertex_kernel,
        dim=ray_origins.shape[0],
        inputs=[
            wp_mesh.id,
            ray_origins,
            ray_directions,
            epsilon,
            num_rays,
            num_triangles,
        ],
        outputs=[output_visible],
        device=ray_origins.device,
    )


def _triangles_visible_from_vertex_cuda_impl(
    mesh_id: np.uint64,
    num_rays: int,
    num_triangles: int,
    total_batches: int,
    vertices: Float[Array, "num_vertices 3"],
    triangles: Int[Array, "num_triangles 3"],
    ray_origins: Float[Array, "total_rays 3"],
    ray_directions: Float[Array, "total_rays 3"],
) -> Bool[Array, "total_batches num_triangles"]:
    return wp.jax_callable(
        _triangles_visible_from_vertex_func,
        output_dims=(total_batches * num_triangles,),
        graph_mode=wp.JaxCallableGraphMode.NONE,
    )(
        mesh_id,
        vertices,
        triangles.ravel(),
        ray_origins,
        ray_directions,
        num_rays,
        num_triangles,
    )[0].reshape((total_batches, num_triangles))


def _triangles_visible_from_vertex_cpu_impl(
    mesh_id: np.uint64,
    num_rays: int,
    num_triangles: int,
    total_batches: int,
    vertices: Float[Array, "num_vertices 3"],
    triangles: Int[Array, "num_triangles 3"],
    ray_origins: Float[Array, "total_rays 3"],
    ray_directions: Float[Array, "total_rays 3"],
) -> Bool[Array, "total_batches num_triangles"]:
    def callback(
        jax_vertices: Float[Array, "num_vertices 3"],
        jax_triangles: Int[Array, "num_triangles 3"],
        jax_ray_origins: Float[Array, "total_rays 3"],
        jax_ray_directions: Float[Array, "total_rays 3"],
    ) -> Bool[Array, " _"]:
        wp_vertices = wp.from_jax(jax_vertices, dtype=wp.vec3)
        wp_triangles = wp.from_jax(jax_triangles.ravel(), dtype=wp.int32)
        wp_ray_origins = wp.from_jax(jax_ray_origins, dtype=wp.vec3)
        wp_ray_directions = wp.from_jax(jax_ray_directions, dtype=wp.vec3)

        output_visible = wp.empty(
            total_batches * num_triangles, dtype=bool, device=wp_ray_origins.device
        )

        _triangles_visible_from_vertex_func(
            int(mesh_id),
            wp_vertices,
            wp_triangles,
            wp_ray_origins,
            wp_ray_directions,
            num_rays,
            num_triangles,
            output_visible,
        )
        return wp.to_jax(output_visible)

    return jax.pure_callback(
        callback,
        jax.ShapeDtypeStruct((total_batches * num_triangles,), bool),
        vertices,
        triangles,
        ray_origins,
        ray_directions,
    ).reshape((total_batches, num_triangles))


_Index = (
    slice
    | Int[ArrayLike, ""]
    | Int[Array, " n"]
    | Bool[Array, " num_triangles"]
    | Sequence[int]
    | Sequence[bool]
)
_T = TypeVar("_T", bound="TriangleMesh")


class _TriangleMeshVerticesUpdateHelper(Generic[_T]):
    """A helper class to update vertices of a triangle mesh."""

    __slots__ = ("mesh",)

    def __init__(self, mesh: _T) -> None:
        self.mesh = mesh

    def __getitem__(self, index: _Index) -> "_TriangleMeshVerticesUpdateRef[_T]":
        return _TriangleMeshVerticesUpdateRef(self.mesh, index)

    def __repr__(self) -> str:
        return f"_TriangleMeshVerticesUpdateHelper({self.mesh!r})"


class _TriangleMeshVerticesUpdateRef(Generic[_T]):
    """A reference to update vertices of a triangle mesh."""

    __slots__ = ("index", "mesh")

    def __init__(self, mesh: _T, index: _Index) -> None:
        if not isinstance(index, slice):
            index_array = jnp.asarray(index)
            if index_array.ndim > 1:
                msg = f"Index must be at most one-dimensional, got array with shape {index_array.shape}."
                raise ValueError(msg)
        self.mesh = mesh
        self.index = index

    def __repr__(self) -> str:
        return f"_TriangleMeshVerticesUpdateRef({self.mesh!r}, {self.index!r})"

    def _triangles_index(self, **kwargs: Unpack[_GetIndexingKwargs]) -> _Index:
        index = self.mesh.triangles.at[self.index, :].get(**kwargs).reshape(-1)
        return jnp.unique(
            index, size=len(index), fill_value=self.mesh.vertices.shape[0]
        )

    def get(
        self, **kwargs: Unpack[_GetIndexingKwargs]
    ) -> Float[ArrayLike, "num_indexed_triangles 3"]:
        # get() is allowed to return duplicates, so we do not use _triangles_index()
        index = self.mesh.triangles.at[self.index, :].get(**kwargs).reshape(-1)
        return self.mesh.vertices.at[index, :].get(wrap_negative_indices=False)

    def set(
        self,
        values: (
            Float[ArrayLike, "3"]
            | Float[ArrayLike, "1"]
            | Float[ArrayLike, ""]
            | Sequence[float]
        ),
        **kwargs: Unpack[_GetIndexingKwargs],
    ) -> _T:
        index = self._triangles_index(**kwargs)
        return eqx.tree_at(
            lambda m: m.vertices,
            self.mesh,
            self.mesh.vertices.at[index, :].set(values, **_AT_INDEXING_KWARGS),
        )

    def add(
        self,
        values: (
            Float[ArrayLike, "3"]
            | Float[ArrayLike, "1"]
            | Float[ArrayLike, ""]
            | Sequence[float]
        ),
        **kwargs: Unpack[_GetIndexingKwargs],
    ) -> _T:
        index = self._triangles_index(**kwargs)
        return eqx.tree_at(
            lambda m: m.vertices,
            self.mesh,
            self.mesh.vertices.at[index, :].add(values, **_AT_INDEXING_KWARGS),
        )

    def sub(
        self,
        values: (
            Float[ArrayLike, "3"]
            | Float[ArrayLike, "1"]
            | Float[ArrayLike, ""]
            | Sequence[float]
        ),
        **kwargs: Unpack[_GetIndexingKwargs],
    ) -> _T:
        index = self._triangles_index(**kwargs)
        return eqx.tree_at(
            lambda m: m.vertices,
            self.mesh,
            self.mesh.vertices.at[index, :].subtract(values, **_AT_INDEXING_KWARGS),
        )

    def mul(
        self,
        values: (
            Float[ArrayLike, "3"]
            | Float[ArrayLike, "1"]
            | Float[ArrayLike, ""]
            | Sequence[float]
        ),
        **kwargs: Unpack[_GetIndexingKwargs],
    ) -> _T:
        index = self._triangles_index(**kwargs)
        return eqx.tree_at(
            lambda m: m.vertices,
            self.mesh,
            self.mesh.vertices.at[index, :].mul(values, **_AT_INDEXING_KWARGS),
        )

    def div(
        self,
        values: (
            Float[ArrayLike, "3"]
            | Float[ArrayLike, "1"]
            | Float[ArrayLike, ""]
            | Sequence[float]
        ),
        **kwargs: Unpack[_GetIndexingKwargs],
    ) -> _T:
        index = self._triangles_index(**kwargs)
        return eqx.tree_at(
            lambda m: m.vertices,
            self.mesh,
            self.mesh.vertices.at[index, :].divide(values, **_AT_INDEXING_KWARGS),
        )

    def pow(
        self,
        values: (
            Float[ArrayLike, "3"]
            | Float[ArrayLike, "1"]
            | Float[ArrayLike, ""]
            | Sequence[float]
        ),
        **kwargs: Unpack[_GetIndexingKwargs],
    ) -> _T:
        index = self._triangles_index(**kwargs)
        return eqx.tree_at(
            lambda m: m.vertices,
            self.mesh,
            self.mesh.vertices.at[index, :].power(values, **_AT_INDEXING_KWARGS),
        )

    def min(
        self,
        values: (
            Float[ArrayLike, "3"]
            | Float[ArrayLike, "1"]
            | Float[ArrayLike, ""]
            | Sequence[float]
        ),
        **kwargs: Unpack[_GetIndexingKwargs],
    ) -> _T:
        index = self._triangles_index(**kwargs)
        return eqx.tree_at(
            lambda m: m.vertices,
            self.mesh,
            self.mesh.vertices.at[index, :].min(values, **_AT_INDEXING_KWARGS),
        )

    def max(
        self,
        values: (
            Float[ArrayLike, "3"]
            | Float[ArrayLike, "1"]
            | Float[ArrayLike, ""]
            | Sequence[float]
        ),
        **kwargs: Unpack[_GetIndexingKwargs],
    ) -> _T:
        index = self._triangles_index(**kwargs)
        return eqx.tree_at(
            lambda m: m.vertices,
            self.mesh,
            self.mesh.vertices.at[index, :].max(values, **_AT_INDEXING_KWARGS),
        )

    def apply(
        self,
        func: Callable[
            [Float[ArrayLike, "num_indexed_triangles 3"]],
            Float[Array, "num_indexed_triangles 3"],
        ],
        **kwargs: Unpack[_GetIndexingKwargs],
    ) -> _T:
        index = self._triangles_index(**kwargs)
        return eqx.tree_at(
            lambda m: m.vertices,
            self.mesh,
            self.mesh.vertices.at[index, :].apply(func, **_AT_INDEXING_KWARGS),
        )


class TriangleMesh(eqx.Module):
    """
    A simple geometry made of triangles.

    .. warning::

        The Warp-accelerated methods in this class (such as :meth:`ray_intersect_any_triangle`,
        :meth:`first_triangle_hit_by_ray`, and :meth:`triangles_visible_from_vertex`)
        only support CPU and CUDA-enabled GPU platforms. They do not support TPUs or other non-CUDA GPUs.
        See :doc:`/limitations` for more details.
    """

    vertices: Float[Array, "num_vertices 3"]
    """The array of triangle vertices."""
    triangles: Int[Array, "num_triangles 3"]
    """The array of triangle indices."""
    face_colors: Float[Array, "num_triangles 3"] | None = eqx.field(default=None)
    """The array of face colors.

    The array contains the face colors, as RGB triplets,
    with a black color used as defaults (if some faces have a color).
    This attribute is :data:`None` if all face colors are unset.
    """
    face_materials: Int[Array, " num_triangles"] | None = eqx.field(default=None)
    """The array of face materials.

    The array contains the material indices,
    with a special placeholder value of ``-1``.
    To obtain the name of the material, see :attr:`material_names`.
    This attribute is :data:`None` if all face materials are unset.
    """
    material_names: tuple[str, ...] = eqx.field(default_factory=tuple, static=True)
    """The list of material names (must be unique)."""
    object_bounds: Int[Array, "num_primitives 2"] | None = eqx.field(default=None)
    """The array of object indices.

    If the present mesh contains multiple objects, usually as a result of appending
    multiple meshes together, this array contain start end end indices for each sub mesh.

    .. important::

        The object indices must cover exactly all triangles in this mesh,
        and be sorted in ascending order. Otherwise, some methods, like
        the random object coloring with :meth:`set_face_colors`, may not
        work as expected.
    """
    assume_quads: bool = eqx.field(default=False)
    """Flag indicating whether triangles can be paired into quadrilaterals.

    Setting this to :data:`True` will not check anything, except that
    :attr:`num_triangles` is even, but each two consecutive
    triangles are assumed to represent a quadrilateral surface.
    """
    assume_unique_vertices: bool = eqx.field(default=False)
    """Flag indicating whether vertices in the mesh are assumed to be unique.

    If set to :data:`False`, methods that rely on unique vertices (like diffraction edge detection)
    will automatically call :meth:`dedup_vertices` first.
    """
    mask: Bool[Array, " num_triangles"] | None = eqx.field(default=None)
    """An optional mask to indicate which triangles are active.

    Using a mask allows to represent multiple sub-meshes of a single mesh, without changing
    the memory allocated to each sub-mesh. Masks can be conveniently created using the
    :meth:`sample` method by passing ``by_masking=True``.

    .. important::

        Unless specified, the transformation or selection operations, like :meth:`rotate`,
        will not take the mask into account, and will apply to all triangles.

    .. important::

        When :attr:`assume_quads` is :data:`True`, a quad is considered active if both triangles
        that form the quad are active.
    """

    def __check_init__(self) -> None:  # noqa: PLW3201
        if self.assume_quads and (self.triangles.shape[0] % 2) != 0:
            msg = "You cannot set 'assume_quads' to 'True' if the number of triangles is not even!"
            raise ValueError(msg)
        if len(set(self.material_names)) != len(self.material_names):
            msg = f"Material names must be unique, got {self.material_names!r}."
            raise ValueError(msg)

    def __del__(self) -> None:
        _WARP_MESHES_CACHE.pop(id(self), None)

    def __getitem__(self, key: slice | Int[ArrayLike, " n"]) -> Self:
        """Return a new instance of this mesh, taking only specific triangles.

        Warning:
            As it is not possible to guarantee that indexing would not break existing
            object bounds, the :attr:`object_bounds` attribute is simply dropped.

        Args:
            key: The key used to index :attr:`triangles`
                along the first axis.

        Returns:
            A new mesh with selected triangles.
        """
        if (
            (isinstance(key, slice) and (key.start is None or key.start == 0))
            and (key.stop is None or key.stop == self.triangles.shape[0])
            and (key.step is None or key.step == 1)
        ):
            return self
        return eqx.tree_at(
            lambda m: (
                m.vertices,
                m.triangles,
                m.face_colors,
                m.face_materials,
                m.object_bounds,
                m.mask,
            ),
            self,
            (
                self.vertices,
                self.triangles[key, :],
                self.face_colors[key, :] if self.face_colors is not None else None,
                self.face_materials[key] if self.face_materials is not None else None,
                None,
                self.mask[key] if self.mask is not None else None,
            ),
            is_leaf=lambda x: x is None,
        )

    def iter_objects(self) -> Iterator[Self]:
        """
        Return an iterator over sub meshes (i.e., objects) defined by :attr:`object_bounds`.

        If :attr:`object_bounds` is :data:`None`, then yield ``self``.

        Yields:
            One or more sub meshes.
        """
        if self.object_bounds is None:
            yield self
        else:
            for start, stop in self.object_bounds:
                yield eqx.tree_at(
                    lambda m: (
                        m.vertices,
                        m.triangles,
                        m.face_colors,
                        m.face_materials,
                        m.object_bounds,
                        m.mask,
                    ),
                    self,
                    (
                        self.vertices,
                        self.triangles.at[start:stop, :].get(
                            **_AT_INDEXING_KWARGS,
                        ),
                        self.face_colors.at[start:stop, :].get(
                            **_AT_INDEXING_KWARGS,
                        )
                        if self.face_colors is not None
                        else None,
                        self.face_materials.at[start:stop].get(
                            **_AT_INDEXING_KWARGS,
                        )
                        if self.face_materials is not None
                        else None,
                        jnp.array([[0, stop - start]]),
                        self.mask.at[start:stop].get(
                            **_AT_INDEXING_KWARGS,
                        )
                        if self.mask is not None
                        else None,
                    ),
                    is_leaf=lambda x: x is None,
                )

    @eqx.filter_jit
    def dedup_vertices(self, num_decimals: int | None = None) -> Self:
        """
        Deduplicate vertices by renumbering triangles to point to the first occurrence.

        This method does not reduce the number of vertices or change their ordering, but
        simply renumbers the triangle indices to refer to the first occurrence of each
        duplicate vertex. Unreferenced vertices will remain in :attr:`vertices`.

        Args:
            num_decimals: The number of decimals to round the vertices to before checking for duplicates.
                If :data:`None`, then no rounding is performed.

        Returns:
            A new mesh with :attr:`assume_unique_vertices` set to :data:`True`.
            If it was already :data:`True`, then this is a no-op and returns ``self``.
        """
        if self.assume_unique_vertices:
            return self

        if self.vertices.shape[0] == 0:
            return eqx.tree_at(lambda m: m.assume_unique_vertices, self, replace=True)

        if num_decimals is None:
            rounded = self.vertices
        else:
            rounded = jnp.round(self.vertices, decimals=num_decimals)

        # Find unique vertices
        _, unique_indices, inverse_indices = jnp.unique(
            rounded,
            axis=0,
            return_index=True,
            return_inverse=True,
            size=self.vertices.shape[0],
        )

        first_occurrence = unique_indices[inverse_indices]
        new_triangles = first_occurrence[self.triangles]

        # Update triangles and set assume_unique_vertices to True
        return eqx.tree_at(
            lambda m: (m.triangles, m.assume_unique_vertices),
            self,
            (new_triangles, True),
        )

    @property
    def num_triangles(self) -> int:
        """The number of triangles."""
        return self.triangles.shape[0]

    @property
    def num_active_triangles(self) -> int | Int[Array, " "]:
        """
        The number of active triangles.

        If :attr:`mask` is not :data:`None`, then the output value can be traced by JAX.
        """
        return jnp.sum(self.mask) if self.mask is not None else self.num_triangles

    @property
    def num_quads(self) -> int:
        """The number of quadrilaterals.

        Raises:
            ValueError: If :attr:`assume_quads` is :data:`False`.
        """
        if not self.assume_quads:
            msg = "Cannot access the number of quadrilaterals if 'assume_quads' is set to 'False'."
            raise ValueError(msg)

        return self.triangles.shape[0] // 2

    @property
    def num_active_quads(self) -> int | Int[Array, " "]:
        """The number of active quadrilaterals.

        If :attr:`mask` is not :data:`None`, then the output value can be traced by JAX.

        Raises:
            ValueError: If :attr:`assume_quads` is :data:`False`.
        """
        if not self.assume_quads:
            msg = "Cannot access the number of active quadrilaterals if 'assume_quads' is set to 'False'."
            raise ValueError(msg)

        return jnp.sum(self.mask[::2]) if self.mask is not None else self.num_quads

    @property
    def num_primitives(self) -> int:
        """The number of primitives.

        This is a convenient alias to :attr:`num_quads` if :attr:`assume_quads` is :data:`True`
        else :attr:`num_triangles`.
        """
        return self.num_quads if self.assume_quads else self.num_triangles

    @property
    def num_active_primitives(self) -> int | Int[Array, " "]:
        """The number of active primitives.

        This is a convenient alias to :attr:`num_active_quads` if :attr:`assume_quads` is :data:`True`
        else :attr:`num_active_triangles`.

        If :attr:`mask` is not :data:`None`, then the output value can be traced by JAX.
        """
        return self.num_active_quads if self.assume_quads else self.num_active_triangles

    @property
    def triangle_vertices(self) -> Float[Array, "num_triangles 3 3"]:
        """The array of indexed triangle vertices."""
        if self.triangles.size == 0:
            return jnp.empty_like(self.vertices, shape=(0, 3, 3))

        return jnp.take(self.vertices, self.triangles, axis=0)

    def set_assume_quads(self, flag: bool = True) -> Self:
        """
        Return a new instance of this scene with :attr:`TriangleMesh.assume_quads<differt.geometry.TriangleMesh.assume_quads>` set to ``flag``.

        Unlike with using :func:`equinox.tree_at`, this function will also
        perform runtime checks.

        Args:
            flag: The new flag value.

        Returns:
            A new mesh with the same structure with :attr:`TriangleMesh.assume_quads<differt.geometry.TriangleMesh.assume_quads>` set to ``flag``.
        """
        mesh = eqx.tree_at(lambda m: m.assume_quads, self, flag)
        mesh.__check_init__()
        return mesh

    @classmethod
    def from_core(cls, core_mesh: differt_core.geometry.TriangleMesh) -> Self:
        """
        Return a triangle mesh from a mesh created by the :mod:`differt_core` module.

        Args:
            core_mesh: The mesh from the core module.

        Returns:
            The corresponding mesh.
        """
        return cls(
            vertices=jnp.asarray(core_mesh.vertices),
            triangles=jnp.asarray(core_mesh.triangles.astype(int)),
            face_colors=jnp.asarray(core_mesh.face_colors)
            if core_mesh.face_colors is not None
            else None,
            face_materials=jnp.asarray(core_mesh.face_materials)
            if core_mesh.face_materials is not None
            else None,
            material_names=tuple(core_mesh.material_names),
            object_bounds=jnp.asarray(core_mesh.object_bounds.astype(int))
            if core_mesh.object_bounds is not None
            else None,
        )

    @property
    def normals(self) -> Float[Array, "num_triangles 3"]:
        """The triangle normals."""
        vectors = jnp.diff(self.triangle_vertices, axis=1)
        normals = jnp.cross(vectors[:, 0, :], vectors[:, 1, :])

        return normalize(normals)[0]

    @property
    def triangle_edges(self) -> Float[Array, "num_triangles 3 2 3"]:
        """The triangle edges."""
        triangle_vertices = self.triangle_vertices
        return jnp.stack(
            (triangle_vertices, jnp.roll(triangle_vertices, 1, axis=-2)), axis=-2
        )

    @eqx.filter_jit
    def _connectivity(
        self,
    ) -> tuple[Int[Array, "num_triangles 3"], Int[Array, "num_triangles 3"]]:
        """
        Compute the edge-to-triangle connectivity of the mesh.

        For each of the 3 edges of each triangle, computes the adjacent triangle index
        and the local edge index of that adjacent triangle.

        Returns:
            A tuple of (adj_t, adj_e):
            - adj_t: Adjacent triangle index (or -1 if none).
            - adj_e: Adjacent edge index (or -1 if none).
        """
        triangles = self.triangles
        assume_quads = self.assume_quads
        num_triangles = triangles.shape[0]
        if num_triangles == 0:
            return jnp.empty((0, 3), dtype=int), jnp.empty((0, 3), dtype=int)

        # For each triangle, the 3 edges are defined by pairs of vertices:
        # Edge 0: vertex 0 -> vertex 2
        # Edge 1: vertex 1 -> vertex 0
        # Edge 2: vertex 2 -> vertex 1
        edges_0 = triangles[:, [0, 2]]
        edges_1 = triangles[:, [1, 0]]
        edges_2 = triangles[:, [2, 1]]
        all_edges = jnp.stack(
            (edges_0, edges_1, edges_2), axis=1
        )  # (num_triangles, 3, 2)

        # Sort the vertices of each edge to make orientation-independent
        sorted_edges = jnp.sort(all_edges, axis=-1)
        flat_edges = sorted_edges.reshape(-1, 2)  # (N, 2)
        n_half_edges = flat_edges.shape[0]

        # Lexicographically sort all half-edges to bring adjacent ones together
        keys = (flat_edges[:, 0], flat_edges[:, 1])
        sorted_indices = jnp.lexsort(keys)
        sorted_flat_edges = flat_edges[sorted_indices]

        # Find adjacent half-edges that are identical (share the same two vertices)
        matches_left = jnp.concatenate([
            jnp.array([False]),
            jnp.all(sorted_flat_edges[1:] == sorted_flat_edges[:-1], axis=-1),
        ])

        # Group identical half-edges
        is_start = ~matches_left
        group_ids = jnp.cumsum(is_start) - 1

        # Count the number of half-edges in each group.
        # A group size of 2 represents a manifold edge (shared by exactly two triangles).
        # A group size greater than 2 represents a non-manifold edge.
        # A group size of 1 is a boundary edge.
        group_counts = jnp.bincount(group_ids, length=n_half_edges)
        element_counts = group_counts[group_ids]

        manifold_count = 2
        is_manifold = element_counts == manifold_count

        # For a manifold pair, match the two half-edges with each other
        match_sorted_idx = jnp.where(
            matches_left, jnp.arange(n_half_edges) - 1, jnp.arange(n_half_edges) + 1
        )
        match_orig_idx = sorted_indices[match_sorted_idx]

        # Initialize adjacency index array with -1
        adj_idx = jnp.full(n_half_edges, -1)
        adj_idx = adj_idx.at[sorted_indices].set(
            jnp.where(is_manifold, match_orig_idx, -1)
        )

        # Convert the matched flat half-edge index back to triangle and local edge indices
        adj_t = jnp.where(adj_idx != -1, adj_idx // 3, -1)
        adj_e = jnp.where(adj_idx != -1, adj_idx % 3, -1)

        adj_t = adj_t.reshape(num_triangles, 3)
        adj_e = adj_e.reshape(num_triangles, 3)

        # Emit warning if any non-manifold edges are found
        def warn_callback(has_non_manifold: Any) -> None:
            if bool(has_non_manifold):
                warnings.warn(
                    "The mesh contains non-manifold edges (edges shared by more than two triangles). "
                    "These edges will be excluded from diffraction calculations.",
                    UserWarning,
                    stacklevel=2,
                )

        jax.debug.callback(warn_callback, jnp.any(element_counts > manifold_count))

        # If assume_quads is True, we filter out the diagonal edges within each quad
        if assume_quads:
            t_idx = jnp.arange(num_triangles)[:, None]
            is_diagonal = jnp.where(
                t_idx % 2 == 0, adj_t == t_idx + 1, adj_t == t_idx - 1
            )
            adj_t = jnp.where(is_diagonal, -1, adj_t)
            adj_e = jnp.where(is_diagonal, -1, adj_e)

        return adj_t, adj_e

    @property
    def diffraction_edges_mask(self) -> Bool[Array, "num_triangles 3"]:
        """The mask to select valid diffraction edges from :attr:`triangle_edges`."""
        if not self.assume_unique_vertices:
            return self.dedup_vertices().diffraction_edges_mask

        num_triangles = self.num_triangles
        if num_triangles == 0:
            return jnp.empty((0, 3), dtype=bool)

        adj_t, _ = self._connectivity()

        # Basic mask: has an adjacent triangle
        mask = adj_t != -1

        # Apply self.mask if present
        if self.mask is not None:
            # Triangle i must be active
            mask = mask & self.mask[:, None]
            # Adjacent triangle must also be active
            adj_t_safe = jnp.where(adj_t != -1, adj_t, num_triangles)
            padded_mask = jnp.append(self.mask, False)
            mask = mask & padded_mask[adj_t_safe]

        # Non-coplanar check
        normals = self.normals
        adj_t_safe = jnp.where(adj_t != -1, adj_t, num_triangles)
        padded_normals = jnp.vstack((normals, jnp.zeros((1, 3))))
        normals_2 = padded_normals[adj_t_safe]
        normals_1 = normals[:, None, :]

        cos_phi = jnp.sum(normals_1 * normals_2, axis=-1)
        is_coplanar = cos_phi > 1.0 - (10.0 * jnp.finfo(cos_phi.dtype).eps)

        return mask & (~is_coplanar)

    def _diffraction_edges_info(
        self,
    ) -> tuple[
        Float[Array, "num_unique_edges 2 3"],
        Int[Array, "num_unique_edges 2"],
        Float[Array, " num_unique_edges"],
    ]:
        """
        Compute coordinates, adjacent triangles, and wedge parameters for unique diffraction edges.

        Returns:
            A tuple of (unique_edges, adj_triangles, wedge_params):
            - unique_edges: Coordinates of unique diffraction edges.
            - adj_triangles: Adjacent triangle indices.
            - wedge_params: Wedge parameters for each unique edge.
        """
        t_idx, e_idx = jnp.where(self.diffraction_edges_mask)
        num_half_edges = t_idx.shape[0]

        # Handle the empty case
        if num_half_edges == 0:
            return (
                jnp.empty((0, 2, 3)),
                jnp.empty((0, 2), dtype=int),
                jnp.empty((0,)),
            )

        v_start = self.triangles[t_idx, e_idx]
        v_end = self.triangles[t_idx, (e_idx - 1) % 3]

        v_min = jnp.minimum(v_start, v_end)
        v_max = jnp.maximum(v_start, v_end)
        keys = jnp.stack((v_min, v_max), axis=-1)

        unique_keys, unique_indices, inverse_indices = jnp.unique(
            keys, axis=0, return_index=True, return_inverse=True
        )
        num_unique_edges = unique_keys.shape[0]

        # Extract coordinates of unique edges
        flat_half_edge_indices = t_idx * 3 + e_idx
        unique_flat_indices = flat_half_edge_indices[unique_indices]
        unique_edges = self.triangle_edges.reshape(-1, 2, 3)[unique_flat_indices]

        # Sort the half-edges by their unique edge index to group them
        sort_idx = jnp.argsort(inverse_indices)
        sorted_inverse = inverse_indices[sort_idx]
        sorted_t_idx = t_idx[sort_idx]

        # Group and assign adjacent triangle indices
        is_second = jnp.concatenate([
            jnp.array([False]),
            sorted_inverse[1:] == sorted_inverse[:-1],
        ])

        adj_triangles = jnp.full((num_unique_edges, 2), -1, dtype=int)

        first_mask = ~is_second
        adj_triangles = adj_triangles.at[sorted_inverse[first_mask], 0].set(
            sorted_t_idx[first_mask]
        )

        second_mask = is_second
        adj_triangles = adj_triangles.at[sorted_inverse[second_mask], 1].set(
            sorted_t_idx[second_mask]
        )

        # Extract wedge parameters for each unique edge
        wedge_params = self.wedge_angles[t_idx[unique_indices], e_idx[unique_indices]]

        return unique_edges, adj_triangles, wedge_params

    @property
    def diffraction_edges(self) -> Float[Array, "num_edges 2 3"]:
        """
        The diffraction edges.

        If you need just-in-time compilation, use :attr:`diffraction_edges_mask` directly.
        """
        if not self.assume_unique_vertices:
            return self.dedup_vertices().diffraction_edges

        edges, _, _ = self._diffraction_edges_info()
        return edges

    @property
    def diffraction_edges_to_triangles(self) -> Int[Array, "num_edges 2"]:
        """
        The indices of the triangles adjacent to each diffraction edge.

        If the edge is isolated (i.e., attached to only one face), the second index is -1.
        """
        if not self.assume_unique_vertices:
            return self.dedup_vertices().diffraction_edges_to_triangles

        _, adj_triangles, _ = self._diffraction_edges_info()
        return adj_triangles

    @property
    def wedge_angles(self) -> Float[Array, "num_triangles 3"]:
        """
        The wedge parameter n (where exterior angle is n * pi) for each edge.

        Returns 1.0 for non-diffraction edges.
        """
        if not self.assume_unique_vertices:
            return self.dedup_vertices().wedge_angles

        num_triangles = self.num_triangles
        if num_triangles == 0:
            return jnp.empty((0, 3))

        normals = self.normals  # (num_triangles, 3)
        adj_t, adj_e = self._connectivity()

        adj_t_safe = jnp.where(adj_t != -1, adj_t, num_triangles)
        padded_normals = jnp.vstack((normals, jnp.zeros((1, 3))))
        normals_2 = padded_normals[adj_t_safe]
        normals_1 = normals[:, None, :]

        cos_phi = jnp.sum(normals_1 * normals_2, axis=-1)
        cos_phi = jnp.clip(cos_phi, -1.0, 1.0)
        phi = jnp.arccos(cos_phi)

        vertices = self.triangle_vertices  # (num_triangles, 3, 3)
        v_a = vertices

        opp_vertex_map = jnp.array([1, 2, 0])
        adj_e_safe = jnp.where(adj_e != -1, adj_e, 0)
        opp_v_idx = opp_vertex_map[adj_e_safe]

        padded_vertices = jnp.vstack((vertices, jnp.zeros((1, 3, 3))))
        v_opp2 = padded_vertices[adj_t_safe, opp_v_idx]

        u2 = v_opp2 - v_a
        dot_u2 = jnp.sum(normals_1 * u2, axis=-1)
        s = jnp.sign(dot_u2)

        n = 1.0 - s * phi / jnp.pi

        mask = self.diffraction_edges_mask
        return jnp.where(mask, n, 1.0)

    @property
    def wedge_parameters(self) -> Float[Array, " num_edges"]:
        """The wedge parameter n for each diffraction edge."""
        if not self.assume_unique_vertices:
            return self.dedup_vertices().wedge_parameters

        _, _, params = self._diffraction_edges_info()
        return params

    @property
    def bounding_box(self) -> Float[Array, "2 3"]:
        """
        The bounding box (min. and max. coordinates).

        .. important::

            Setting :attr:`mask` will have an effect on the bounding box,
            as the bounding box is computed only for the active triangles.
        """
        # Using self.triangle_vertices is important because, e.g., as a result of using
        # __getitem__, some vertices in 'self.vertices' may no longer be used by this mesh.
        vertices = self.triangle_vertices
        where = self.mask[:, None, None] if self.mask is not None else None
        return jnp.vstack(
            (
                jnp.min(vertices, axis=(0, 1), initial=+jnp.inf, where=where),
                jnp.max(vertices, axis=(0, 1), initial=-jnp.inf, where=where),
            ),
        )

    if TYPE_CHECKING:

        @property
        def at(self) -> _TriangleMeshVerticesUpdateHelper[Self]: ...

    @property
    def at(self):  # noqa: ANN202
        """
        Helper property for updating or indexing a subset of triangle vertices.

        This ``at`` property is used to update vertices of a triangle mesh,
        based on triangles indices,
        similar to how the ``at`` property is used in :attr:`jax.numpy.ndarray.at`.

        In particular, the following methods are available:

        - ``get(**kwargs)``: Get the vertices of selected triangles;
        - ``set(values, **kwargs)``: Set the vertices of selected triangles to some values;
        - ``add(values, **kwargs)``: Add some values to the vertices of selected triangles;
        - ``sub(values, **kwargs)``: Subtract some values from the vertices of selected triangles;
        - ``mul(values, **kwargs)``: Multiply the vertices of selected triangles by some values;
        - ``div(values, **kwargs)``: Divide the vertices of selected triangles by some values;
        - ``pow(values, **kwargs)``: Raise the vertices of selected triangles to some power;
        - ``min(values, **kwargs)``: Take the element-wise minimum with the vertices of selected triangles;
        - ``max(values, **kwargs)``: Take the element-wise maximum with the vertices of selected triangles;
        - ``apply(func, **kwargs)``: Apply a function to the vertices of selected triangles.

        E.g., ``mesh.at[0:2].add([1.0, 2.0, 3.0])`` will translate the first two triangles.

        Each method accepts the same keyword arguments as the corresponding ``get`` method of
        :attr:`jax.numpy.ndarray.at`.
        These keyword arguments control how triangle indices are resolved internally.

        Because the vertices of a triangle mesh may be shared
        between multiple triangles, this method prevents updating the same vertex multiple times
        by ignoring duplicate vertex indices. As a result, providing duplicate triangle indices
        will not result in duplicate updates.

        Warning:
            As duplicate vertices are ignored, the number of update vertices is not
            necessarily equal to the number of triangles selected times three. Moreover,
            vertices are re-ordered when duplicates are removed. As a results, you should
            not apply any update that depends on the order of or the number of updated
            the vertices.

        Examples:
            The following example shows how to translate the first two (triangle) faces.

            .. plotly::
                :context: reset

                >>> from differt.geometry import TriangleMesh
                >>>
                >>> mesh = (
                ...     sum(
                ...         TriangleMesh.box().iter_objects(),
                ...         start=TriangleMesh.empty(),
                ...     )
                ...     .at[0:2]
                ...     .add([1.0, 1.0, 0.0])
                ... )
                >>> fig = mesh.plot(opacity=0.5, backend="plotly")
                >>> fig  # doctest: +SKIP

            In the above example, splitting the cube mesh into separate objects is necessary,
            as the vertices of the cube are shared between the faces. If the cube was not split,
            the translation would be applied to all the faces that share the vertices of the first two faces.

            .. plotly::
                :context:

                >>> mesh = TriangleMesh.box().at[0:2].add([1.0, 1.0, 0.0])
                >>> fig = mesh.plot(opacity=0.5, backend="plotly")
                >>> fig  # doctest: +SKIP

            Finally, the :attr:`at` property is lazily evaluated, so checking that the index
            is valid is not performed until a method is called.

            >>> from differt.geometry import TriangleMesh
            >>>
            >>> mesh = TriangleMesh.box()
            >>> mesh.at
            _TriangleMeshVerticesUpdateHelper(TriangleMesh(
              vertices=f32[8,3],
              triangles=i32[10,3],
              material_names=(),
              object_bounds=i32[5,2],
              assume_unique_vertices=True
            ))
            >>> index = jnp.array([True, False])
            >>> mesh.at[index]
            _TriangleMeshVerticesUpdateRef(TriangleMesh(
              vertices=f32[8,3],
              triangles=i32[10,3],
              material_names=(),
              object_bounds=i32[5,2],
              assume_unique_vertices=True
            ), Array([ True, False], dtype=bool))
            >>> mesh.at[index].add(1.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            IndexError: boolean index did not match shape of indexed array in index 0:
            got (2,), expected (10,)
        """
        return _TriangleMeshVerticesUpdateHelper(self)

    def masked(self) -> Self:
        """
        Return a new instance of this object that only keeps masked (i.e., active) triangles.

        .. important::
            This method does not preserve the :attr:`object_bounds` attribute.

        Returns:
            A new paths instance with flattened batch dimensions and only valid paths.
        """
        if self.mask is None:
            return jax.tree.map(lambda m: m, self)

        triangles = self.triangles[self.mask, :]
        # We will need to re-index the vertices, so we first get the unique vertex indices used by the active triangles,
        used_indices = jnp.unique(triangles.reshape(-1))
        # Then we create a lookup table to re-index the triangles, where the indices of the used vertices are replaced by their new indices.
        lookup = jnp.empty((self.vertices.shape[0],), dtype=triangles.dtype)
        lookup = lookup.at[used_indices].set(
            jnp.arange(used_indices.shape[0], dtype=triangles.dtype)
        )
        # Keep unique vertices
        vertices = self.vertices[used_indices, :]
        triangles = lookup[triangles]

        return eqx.tree_at(
            lambda m: (
                m.vertices,
                m.triangles,
                m.face_colors,
                m.face_materials,
                m.object_bounds,
                m.mask,
            ),
            self,
            (
                vertices,
                triangles,
                self.face_colors[self.mask, :]
                if self.face_colors is not None
                else None,
                self.face_materials[self.mask]
                if self.face_materials is not None
                else None,
                None,
                None,
            ),
            is_leaf=lambda x: x is None,
        )

    def rotate(self, rotation_matrix: Float[ArrayLike, "3 3"]) -> Self:
        """
        Return a new mesh by applying a rotation matrix to all triangle coordinates.

        Args:
            rotation_matrix: The rotation matrix.

        Returns:
            The new rotated mesh.
        """
        return eqx.tree_at(
            lambda m: m.vertices,
            self,
            (jnp.asarray(rotation_matrix) @ self.vertices.T).T,
        )

    def scale(self, scale_factor: Float[ArrayLike, ""]) -> Self:
        """
        Return a new mesh by applying a scale factor to all triangle coordinates.

        Args:
            scale_factor: The scale factor.

        Returns:
            The new scaled mesh.
        """
        return eqx.tree_at(
            lambda m: m.vertices,
            self,
            self.vertices * scale_factor,
        )

    def translate(self, translation: Float[ArrayLike, "3"]) -> Self:
        """
        Return a new mesh by applying a translation to all triangle coordinates.

        Args:
            translation: The translation vector.

        Returns:
            The new translated mesh.
        """
        return eqx.tree_at(
            lambda m: m.vertices,
            self,
            self.vertices + jnp.asarray(translation),
        )

    def clip(
        self,
        x_min: Float[ArrayLike, ""] | None = None,
        x_max: Float[ArrayLike, ""] | None = None,
        y_min: Float[ArrayLike, ""] | None = None,
        y_max: Float[ArrayLike, ""] | None = None,
        z_min: Float[ArrayLike, ""] | None = None,
        z_max: Float[ArrayLike, ""] | None = None,
    ) -> Self:
        """
        Clip all vertex coordinates to the given bounds.

        Args:
            x_min: The minimum x coordinate.
            x_max: The maximum x coordinate.
            y_min: The minimum y coordinate.
            y_max: The maximum y coordinate.
            z_min: The minimum z coordinate.
            z_max: The maximum z coordinate.

        Returns:
            A new mesh with clipped vertices.

        Examples:
            The following example shows the effect on a plotted mesh before and after clipping.

            .. plotly::
                :context: reset

                >>> from differt.geometry import TriangleMesh
                >>>
                >>> mesh = TriangleMesh.box(length=4.0, width=2.0, height=1.0)
                >>> fig = mesh.plot(backend="plotly")
                >>> fig  # doctest: +SKIP

            .. plotly::
                :context:

                >>> clipped = mesh.clip(x_min=-1.0, x_max=1.0, y_min=-0.5, y_max=0.5)
                >>> fig = clipped.plot(backend="plotly")
                >>> fig  # doctest: +SKIP
        """
        vertices = self.vertices

        if x_min is not None or x_max is not None:
            vertices = vertices.at[:, 0].apply(
                lambda x: jnp.clip(x, min=x_min, max=x_max)
            )
        if y_min is not None or y_max is not None:
            vertices = vertices.at[:, 1].apply(
                lambda y: jnp.clip(y, min=y_min, max=y_max)
            )
        if z_min is not None or z_max is not None:
            vertices = vertices.at[:, 2].apply(
                lambda z: jnp.clip(z, min=z_min, max=z_max)
            )

        return eqx.tree_at(lambda m: m.vertices, self, vertices)

    @classmethod
    def empty(cls) -> Self:
        """
        Create a empty mesh.

        Returns:
            A new empty scene.
        """
        return cls(
            vertices=jnp.empty((0, 3)),
            triangles=jnp.empty((0, 3), dtype=int),
            assume_unique_vertices=True,
        )

    def append(self, other: "TriangleMesh") -> Self:
        """
        Return a new mesh by appending another mesh to this one.

        .. tip::

            For convenience, you can also use the ``+`` operator.

        .. note::

            The following rules are applied when merging two meshes:

            * The vertices are concatenated;
            * The triangles are concatenated, and the indices of the second mesh are updated;
            * The face colors are concatenated. If one mesh has colors while the other does not,
              then mesh with no colors will have its face colors set to black (0, 0, 0);
            * The face materials are concatenated. If ``other`` has face materials not included
              in ``self``, then the face materials from ``other`` are renumbered.
              If one mesh has colors while the other does not,
              then mesh with no colors will have its face materials set to ``-1``;
            * The material names are merged, keeping only unique names;
            * The object bounds are concatenated only if both meshes have them set,
              otherwise, the object bounds are set to :data:`None`;
            * The masks are concatenated if present in both meshes. If one mesh has a mask while the other does not,
              then the mesh with no mask will have its mask set to all triangles being active (i.e., :data:`True`).
            * The :attr:`assume_quads` flag is set to :data:`True` if both meshes have it set to :data:`True`.

            Two important exceptions are:

            * If one mesh is empty, a new instance of the other mesh is returned as is;
            * If both meshes are empty, then a new instance of ``self`` is returned.

        Args:
            other: The mesh to append.

        Returns:
            The new mesh with a structure that is the result of combining both input meshes.

        Examples:
            The following example shows how to create a mesh of nested cubes.

            .. plotly::

                >>> from differt.geometry import TriangleMesh
                >>>
                >>> mesh = TriangleMesh.empty()
                >>> for i in range(3):
                ...     size = 1.0 / (i + 1)
                ...     mesh += TriangleMesh.box(length=size, width=size, height=size)
                >>> mesh = mesh.set_assume_quads().set_face_colors(
                ...     key=jax.random.key(1234)
                ... )
                >>> fig = mesh.plot(opacity=0.5, backend="plotly")
                >>> fig  # doctest: +SKIP
        """
        if other.is_empty:
            return eqx.tree_at(lambda _: (), self, ())
        if self.is_empty:
            return eqx.tree_at(lambda _: (), other, ())

        vertices = jnp.concatenate((self.vertices, other.vertices), axis=0)
        triangles = jnp.concatenate(
            (self.triangles, other.triangles + self.vertices.shape[0]), axis=0
        )

        if self.face_colors is not None and other.face_colors is not None:
            face_colors = jnp.concatenate((self.face_colors, other.face_colors), axis=0)
        elif self.face_colors is not None:
            face_colors = jnp.concatenate(
                (
                    self.face_colors,
                    jnp.zeros_like(self.face_colors, shape=(other.num_triangles, 3)),
                ),
                axis=0,
            )
        elif other.face_colors is not None:
            face_colors = jnp.concatenate(
                (
                    jnp.zeros_like(other.face_colors, shape=(self.num_triangles, 3)),
                    other.face_colors,
                ),
                axis=0,
            )
        else:
            face_colors = None

        material_names = dict.fromkeys(self.material_names) | dict.fromkeys(
            other.material_names
        )
        material_indices = {
            material_name: i for i, material_name in enumerate(material_names)
        }
        other_face_materials_renumbered = jnp.array([
            material_indices[name] for name in other.material_names
        ])
        material_names = tuple(material_names)

        if self.face_materials is not None and other.face_materials is not None:
            face_materials = jnp.concatenate(
                (
                    self.face_materials,
                    jnp.where(
                        other.face_materials != -1,
                        other_face_materials_renumbered[other.face_materials],
                        other.face_materials,
                    ),
                ),
                axis=0,
            )
        elif self.face_materials is not None:
            face_materials = jnp.concatenate(
                (
                    self.face_materials,
                    jnp.full_like(self.face_materials, -1, shape=other.num_triangles),
                ),
                axis=0,
            )
        elif other.face_materials is not None:
            face_materials = jnp.concatenate(
                (
                    jnp.full_like(other.face_materials, -1, shape=self.num_triangles),
                    jnp.where(
                        other.face_materials != -1,
                        other_face_materials_renumbered[other.face_materials],
                        other.face_materials,
                    ),
                ),
                axis=0,
            )
        else:
            face_materials = None

        object_bounds = (
            jnp.concatenate(
                (self.object_bounds, other.object_bounds + self.num_triangles), axis=0
            )
            if (self.object_bounds is not None and other.object_bounds is not None)
            else None
        )

        if self.mask is not None and other.mask is not None:
            mask = jnp.concatenate((self.mask, other.mask), axis=0)
        elif self.mask is not None:
            mask = jnp.concatenate(
                (self.mask, jnp.ones_like(self.mask, shape=(other.num_triangles,))),
                axis=0,
            )
        elif other.mask is not None:
            mask = jnp.concatenate(
                (
                    jnp.ones_like(other.mask, shape=(self.num_triangles,)),
                    other.mask,
                ),
                axis=0,
            )
        else:
            mask = None

        assume_quads = self.assume_quads and other.assume_quads
        mesh = replace(
            self,
            material_names=material_names,
            assume_quads=assume_quads,
            assume_unique_vertices=False,
        )
        return eqx.tree_at(
            lambda m: (
                m.vertices,
                m.triangles,
                m.face_colors,
                m.face_materials,
                m.object_bounds,
                m.mask,
            ),
            mesh,
            (vertices, triangles, face_colors, face_materials, object_bounds, mask),
            is_leaf=lambda x: x is None,
        )

    __add__ = append

    def drop_unused_vertices(self) -> Self:
        """
        Return a new mesh with unused vertices (not referenced by any triangle) removed.

        Returns:
            A new mesh with unused vertices removed.
        """
        if self.vertices.shape[0] == 0:
            return self

        unique_referenced = jnp.unique(self.triangles)
        new_vertices = self.vertices[unique_referenced]
        new_triangles = jnp.searchsorted(unique_referenced, self.triangles)

        return eqx.tree_at(
            lambda m: (m.vertices, m.triangles),
            self,
            (new_vertices, new_triangles),
        )

    def drop_duplicates(self) -> Self:
        """
        Return a new mesh with duplicate vertices removed.

        This method first deduplicates the vertices by calling :meth:`dedup_vertices`
        to renumber triangle indices, and then removes any unused vertices by calling
        :meth:`drop_unused_vertices`.

        Returns:
            A new mesh with duplicate vertices removed.
        """
        return self.dedup_vertices().drop_unused_vertices()

    @overload
    def set_face_colors(
        self,
        colors: Float[ArrayLike, "#num_triangles 3"] | Float[ArrayLike, "3"],
        *,
        key: None = ...,
    ) -> Self: ...

    @overload
    def set_face_colors(
        self,
        colors: None = ...,
        *,
        key: PRNGKeyArray,
    ) -> Self: ...

    def set_face_colors(
        self,
        colors: Float[ArrayLike, "#num_triangles 3"]
        | Float[ArrayLike, "3"]
        | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Self:
        """
        Return a new instance of this mesh, with new face colors.

        Args:
            colors: The array of RGB colors.
                If one color is provided, it will be applied to all triangles.

                This or ``key`` must be specified.
            key: If provided, colors will be randomly generated.

                If :attr:`object_bounds` is not :data:`None`, then triangles
                within the same object will share the same color. Otherwise,
                a random color is generated for each triangle
                (or quadrilateral if :attr:`assume_quads` is :data:`True`).

        Returns:
            A new mesh with updated face colors.

        Raises:
            ValueError: If ``colors`` or ``key`` is not specified.

        Examples:
            The following example shows how this function paints the mesh,
            for different argument types.

            First, we load a scene from Sionna :cite:`sionna`, that is
            already colored, and extract the mesh from it.

            .. plotly::
                :context: reset

                >>> from differt.scene import (
                ...     TriangleScene,
                ...     download_sionna_scenes,
                ...     get_sionna_scene,
                ... )
                >>>
                >>> download_sionna_scenes()  # doctest: +SKIP
                >>> file = get_sionna_scene("simple_street_canyon")
                >>> mesh = TriangleScene.load_xml(file).mesh
                >>> fig = mesh.plot(backend="plotly")
                >>> fig  # doctest: +SKIP

            Then, we could set the same color to all triangles.

            .. plotly::
                :context:

                >>> fig = mesh.set_face_colors(jnp.array([0.8, 0.2, 0.0])).plot(
                ...     backend="plotly"
                ... )
                >>> fig  # doctest: +SKIP

            We could also manually specify a different color for each triangle, but it can
            become tedious as the number of triangles gets larger. Another option is to rely
            on automatic random coloring, using the ``key`` argument.

            As our mesh is a collection of 7 distinct objects, as this was loaded from a Sionna
            XML file, this utility will automatically detect it and color each
            object differently.

            .. plotly::
                :context:

                >>> mesh.object_bounds
                Array([[ 0, 12],
                       [12, 24],
                       [24, 36],
                       [36, 48],
                       [48, 60],
                       [60, 72],
                       [72, 74]], dtype=int32)
                >>> fig = mesh.set_face_colors(key=jax.random.key(1234)).plot(
                ...     backend="plotly"
                ... )
                >>> fig  # doctest: +SKIP

            If you prefer to have per-triangle coloring, you can perform surgery on the mesh
            to remove its :attr:`object_bounds` attribute.

            .. plotly::
                :context:

                >>> import equinox as eqx
                >>>
                >>> mesh = eqx.tree_at(lambda m: m.object_bounds, mesh, None)
                >>> fig = mesh.set_face_colors(key=jax.random.key(1234)).plot(
                ...     backend="plotly"
                ... )
                >>> fig  # doctest: +SKIP

            Finally, you can also set :attr:`assume_quads` to :data:`True` to color quadrilaterals
            instead.

            .. plotly::
                :context:

                >>> fig = (
                ...     mesh
                ...     .set_assume_quads()
                ...     .set_face_colors(key=jax.random.key(1234))
                ...     .plot(backend="plotly")
                ... )
                >>> fig  # doctest: +SKIP
        """
        if (colors is None) == (key is None):
            msg = "You must specify one of 'colors' or `key`, not both."
            raise ValueError(msg)

        if key is not None:
            if self.object_bounds is not None:
                object_colors = jax.random.uniform(
                    key, (self.object_bounds.shape[0], 3)
                )
                repeats = jnp.diff(self.object_bounds, axis=-1)
                colors = jnp.repeat(
                    object_colors,
                    repeats,
                    axis=0,
                    total_repeat_length=self.num_triangles,
                )
            elif self.assume_quads:
                quad_colors = jax.random.uniform(key, (self.num_quads, 3))
                repeats = jnp.full(self.num_primitives, 2)
                colors = jnp.repeat(
                    quad_colors, repeats, axis=0, total_repeat_length=self.num_triangles
                )
            else:
                colors = jax.random.uniform(key, (self.num_triangles, 3))

            return self.set_face_colors(colors=colors)

        face_colors = jnp.broadcast_to(
            jnp.asarray(colors).reshape(-1, 3),
            self.triangles.shape,
        )
        return eqx.tree_at(
            lambda m: m.face_colors,
            self,
            face_colors,
            is_leaf=lambda x: x is None,
        )

    def set_materials(self, *names: str) -> Self:
        """
        Return a new instance of this mesh, with new face materials from material names.

        If a material name is not in :attr:`material_names`, it is added.

        Args:
            names: The material names.
                If one name is provided, it will be applied to all triangles.

        Returns:
            A new mesh with updated face materials.

        Raises:
            ValueError: If the number of names is not 1, :attr:`num_triangles`, or :attr:`num_primitives` (if :attr:`assume_quads` is set to :data:`True`).
        """
        if len(names) not in {1, self.num_triangles, self.num_primitives}:
            if self.assume_quads:
                msg = f"Expected either 1, {self.num_triangles}, or {self.num_primitives} names, got {len(names)}."
            else:
                msg = f"Expected either 1, or {self.num_triangles} names, got {len(names)}."
            raise ValueError(msg)
        material_names = dict.fromkeys(self.material_names)
        if all(name in material_names for name in names):
            material_names = {name: i for i, name in enumerate(material_names)}
            face_materials = jnp.array([material_names[name] for name in names])
            if self.assume_quads and len(names) == self.num_quads:
                face_materials = jnp.repeat(face_materials, 2)
            return self.set_face_materials(face_materials)

        material_names = material_names | dict.fromkeys(names)
        material_names = {name: i for i, name in enumerate(material_names)}

        face_materials = jnp.array([material_names[name] for name in names])
        if self.assume_quads and len(names) == self.num_quads:
            face_materials = jnp.repeat(face_materials, 2)
        mesh = replace(self, material_names=tuple(material_names))
        return mesh.set_face_materials(face_materials)

    def set_face_materials(
        self, materials: Int[ArrayLike, ""] | Int[ArrayLike, "#num_triangles"]
    ) -> Self:
        """
        Return a new instance of this mesh, with new face materials.

        Args:
            materials: The material indices.
                If one material is provided, it will be applied to all triangles.

                No check is performed to verify that material indices are actually
                in bounds of :attr:`material_names`.

        Returns:
            A new mesh with updated face materials.
        """
        face_materials = jnp.broadcast_to(
            jnp.asarray(materials),
            self.num_triangles,
        )
        return eqx.tree_at(
            lambda m: m.face_materials,
            self,
            face_materials,
            is_leaf=lambda x: x is None,
        )

    @overload
    @classmethod
    def plane(
        cls,
        vertex_a: Float[ArrayLike, "3"],
        vertex_b: Float[ArrayLike, "3"],
        vertex_c: Float[ArrayLike, "3"],
        *,
        normal: None = ...,
        side_length: Float[ArrayLike, ""] = ...,
        rotate: Float[ArrayLike, ""] | None = ...,
    ) -> Self: ...

    @overload
    @classmethod
    def plane(
        cls,
        vertex_a: Float[ArrayLike, "3"],
        vertex_b: None = ...,
        vertex_c: None = ...,
        *,
        normal: Float[ArrayLike, "3"],
        side_length: Float[ArrayLike, ""] = ...,
        rotate: Float[ArrayLike, ""] | None = ...,
    ) -> Self: ...

    @classmethod
    def plane(
        cls,
        vertex_a: Float[ArrayLike, "3"],
        vertex_b: Float[ArrayLike, "3"] | None = None,
        vertex_c: Float[ArrayLike, "3"] | None = None,
        *,
        normal: Float[ArrayLike, "3"] | None = None,
        side_length: Float[ArrayLike, ""] = 1.0,
        rotate: Float[ArrayLike, ""] | None = None,
    ) -> Self:
        """
        Create a plane mesh, made of two triangles.

        Note:
            The mesh satisfies the guarantees
            expected when setting
            :attr:`assume_quads` to :data:`True`.

        Args:
            vertex_a: The center of the plane.
            vertex_b: Any second vertex on the plane.

                This and ``vertex_c``, or ``normal`` is required.
            vertex_c: Any third vertex on the plane.

                This and ``vertex_b``, or ``normal`` is required.
            normal: The plane normal.

                Must be of unit length.
            side_length: The side length of the plane.
            rotate: An optional rotation angle, in radians,
                to be applied around the normal of the plane
                and its center.

        Returns:
            A new plane mesh.

        Raises:
            ValueError: If neither ``vertex_b`` and ``vertex_c``, nor ``normal`` have been provided,
                or if both have been provided simultaneously.
        """
        if (vertex_b is None) != (vertex_c is None):
            msg = "You must specify either of both  of 'vertex_b' and 'vertex_c', or none."
            raise ValueError(msg)

        if (vertex_b is None) == (normal is None):
            msg = "You must specify one of ('vertex_b', 'vertex_c') or 'normal', not both."
            raise ValueError(msg)

        vertex_a = jnp.asarray(vertex_a)

        if vertex_b is not None:
            vertex_b = jnp.asarray(vertex_b)
            vertex_c = jnp.asarray(vertex_c)
            u = vertex_b - vertex_a
            v = vertex_c - vertex_a
            w = jnp.cross(u, v)
            normal = normalize(w)[0]
        else:
            normal = jnp.asarray(normal)

        u, v = orthogonal_basis(
            normal,
        )

        s = 0.5 * side_length

        vertices = s * jnp.array([u + v, v - u, -u - v, u - v])

        if rotate:
            rotation_matrix = rotation_matrix_along_axis(rotate, normal)
            vertices = (rotation_matrix @ vertices.T).T

        vertices += vertex_a

        triangles = jnp.array([[0, 1, 2], [0, 2, 3]], dtype=int)
        return cls(vertices=vertices, triangles=triangles, assume_unique_vertices=True)

    @classmethod
    def box(
        cls,
        length: Float[ArrayLike, ""] = 1.0,
        width: Float[ArrayLike, ""] = 1.0,
        height: Float[ArrayLike, ""] = 1.0,
        *,
        with_top: bool = False,
        with_bottom: bool = True,
    ) -> Self:
        """
        Create a box mesh, with an optional opening on the top.

        Note:
            The mesh satisfies the guarantees
            expected when setting
            :attr:`assume_quads` to :data:`True`.

        Args:
            length: The length of the box (along x-axis).
            width: The width of the box (along y-axis).
            height: The height of the box (along z-axis).
            with_top: Whether the top of part
                of the box is included or not.
            with_bottom: Whether the bottom of part
                of the box is included or not.

        Returns:
            A new box mesh.

        Examples:
            The following example shows how to create a cube.

            .. plotly::
                :context: reset

                >>> from differt.geometry import TriangleMesh
                >>>
                >>> mesh = (
                ...     TriangleMesh
                ...     .box(with_top=True)
                ...     .set_assume_quads()
                ...     .set_face_colors(key=jax.random.key(1234))
                ... )
                >>> fig = mesh.plot(opacity=0.5, backend="plotly")
                >>> fig  # doctest: +SKIP

            The second example shows how to create a corridor-like
            mesh, without the ceiling face.

            .. plotly::
                :context:

                >>> mesh = (
                ...     TriangleMesh
                ...     .box(length=10.0, width=3.0, height=2.0)
                ...     .set_assume_quads()
                ...     .set_face_colors(key=jax.random.key(1234))
                ... )
                >>> fig = mesh.plot(opacity=0.5, backend="plotly")
                >>> fig = fig.update_scenes(aspectmode="data")
                >>> fig  # doctest: +SKIP
        """
        dx = jnp.array([length * 0.5, 0.0, 0.0])
        dy = jnp.array([0.0, width * 0.5, 0.0])
        dz = jnp.array([0.0, 0.0, height * 0.5])

        vertices = jnp.stack((
            +dx + dy + dz,
            +dx + dy - dz,
            -dx + dy - dz,
            -dx + dy + dz,
            -dx - dy - dz,
            -dx - dy + dz,
            +dx - dy - dz,
            +dx - dy + dz,
        ))
        triangles = jnp.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [3, 2, 4],
                [3, 4, 5],
                [5, 4, 6],
                [5, 6, 7],
                [7, 6, 1],
                [7, 1, 0],
            ],
            dtype=int,
        )
        if with_bottom:
            triangles = jnp.concatenate(
                (triangles, jnp.asarray([[1, 4, 2], [1, 6, 4]])),
                axis=0,
            )
        if with_top:
            triangles = jnp.concatenate(
                (triangles, jnp.asarray([[0, 3, 5], [0, 5, 7]])),
                axis=0,
            )

        indices = jnp.arange(0, triangles.shape[0] + 1, 2)
        object_bounds = jnp.column_stack((indices[:-1], indices[+1:]))
        return cls(
            vertices=vertices,
            triangles=triangles,
            object_bounds=object_bounds,
            assume_unique_vertices=True,
        )

    @property
    def is_empty(self) -> bool:
        """Whether this scene has no triangle."""
        return self.triangles.size == 0

    @classmethod
    def load_obj(cls, file: str) -> Self:
        """
        Load a triangle mesh from a Wavefront .obj file.

        Currently, only vertices and triangles are loaded. Triangle normals
        are computed afterward (when first accessed).

        Args:
            file: The path to the Wavefront .obj file.

        Returns:
            The corresponding mesh containing only triangles.
        """
        core_mesh = differt_core.geometry.TriangleMesh.load_obj(file)
        return cls.from_core(core_mesh)

    @classmethod
    def load_ply(cls, file: str) -> Self:
        """
        Load a triangle mesh from a Stanford PLY .ply file.

        Currently, only vertices and triangles are loaded. Triangle normals
        are computed afterward (when first accessed).

        Args:
            file: The path to the Stanford PLY .ply file.

        Returns:
            The corresponding mesh containing only triangles.
        """
        core_mesh = differt_core.geometry.TriangleMesh.load_ply(file)
        return cls.from_core(core_mesh)

    def plot(
        self,
        *,
        show_normals: bool = False,
        show_triangle_edges: bool = False,
        show_diffraction_edges: bool = False,
        normals_kwargs: Mapping[str, Any] | None = None,
        triangle_edges_kwargs: Mapping[str, Any] | None = None,
        diffraction_edges_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> PlotOutput:
        """
        Plot this mesh on a 3D scene.

        Args:
            show_normals: Whether to show the normals of the triangles.
            show_triangle_edges: Whether to show the edges of the triangles.
            show_diffraction_edges: Whether to show the diffraction edges.
            normals_kwargs: A mapping of keyword arguments passed to
                :func:`draw_rays<differt.plotting.draw_rays>`.
            triangle_edges_kwargs: A mapping of keyword arguments passed to
                :func:`draw_paths<differt.plotting.draw_paths>`.
            diffraction_edges_kwargs: A mapping of keyword arguments passed to
                :func:`draw_paths<differt.plotting.draw_paths>`.
            kwargs: Keyword arguments passed to
                :func:`draw_mesh<differt.plotting.draw_mesh>`.

        Returns:
            The resulting plot output.

        Examples:
            The following examples show how to customize the plotting of a cube.

            First, we plot the mesh with a default color.

            .. plotly::
                :context: reset

                >>> from differt.geometry import TriangleMesh
                >>> mesh = TriangleMesh.box(with_top=True)
                >>> fig = mesh.plot(opacity=0.5, backend="plotly")
                >>> fig  # doctest: +SKIP

            Next, we plot the mesh, but with normals and triangle edges.

            .. plotly::
                :context:

                >>> fig = mesh.plot(
                ...     show_normals=True,
                ...     show_triangle_edges=True,
                ...     normals_kwargs={"name": "normals", "color": "red"},
                ...     triangle_edges_kwargs={
                ...         "name": "triangle edges",
                ...         "line_color": "yellow",
                ...     },
                ...     opacity=0.5,
                ...     backend="plotly",
                ... )
                >>> fig  # doctest: +SKIP

            Finally, we plot the mesh with diffraction edges. Diffraction edges are relatively expensive to compute,
            as they are computed by first computing the triangle edges, then removing duplicates, and removing co-planar edges.

            .. plotly::
                :context:

                >>> fig = mesh.plot(
                ...     show_diffraction_edges=True,
                ...     diffraction_edges_kwargs={
                ...         "name": "diffraction edges",
                ...         "line_color": "yellow",
                ...     },
                ...     opacity=0.5,
                ...     backend="plotly",
                ... )
                >>> fig  # doctest: +SKIP
        """
        mesh = self.masked()
        if "face_colors" not in kwargs and mesh.face_colors is not None:
            kwargs["face_colors"] = mesh.face_colors

        normals_kwargs = {} if normals_kwargs is None else normals_kwargs
        triangle_edges_kwargs = (
            {} if triangle_edges_kwargs is None else triangle_edges_kwargs
        )
        diffraction_edges_kwargs = (
            {} if diffraction_edges_kwargs is None else diffraction_edges_kwargs
        )

        with reuse(pass_all_kwargs=False, **kwargs) as result:
            draw_mesh(
                vertices=mesh.vertices,
                triangles=mesh.triangles,
                **kwargs,
            )

            if show_normals:
                draw_rays(
                    mesh.triangle_vertices.mean(axis=-2),
                    mesh.normals,
                    **normals_kwargs,
                )

            if show_triangle_edges:
                draw_paths(
                    mesh.triangle_edges,
                    **triangle_edges_kwargs,
                )
            if show_diffraction_edges:
                draw_paths(
                    mesh.diffraction_edges,
                    **diffraction_edges_kwargs,
                )

        return result

    def sample(
        self,
        size: int | Float[ArrayLike, ""],
        replace: bool = False,
        preserve: bool = False,
        *,
        by_masking: bool = False,
        sample_objects: bool = False,
        key: PRNGKeyArray,
    ) -> Self:
        """
        Generate a new mesh by randomly sampling primitives from this geometry.

        Important:
            If ``by_masking`` is set to :data:`True`, then this function is compatible
            with :func:`jax.jit`. Conversely, if ``by_masking`` is set to :data:`False`,
            then this function is **not** compatible with :func:`jax.jit`,
            as the size of the output mesh is not known at compile time.

        Args:
            size: The size of the sample, i.e., the number of primitives (or objects, if ``sample_objects`` is :data:`True`).

                If a floating point number is provided, it is interpreted as a fill factor,
                i.e., the fraction of primitives to sample from the mesh. This is only supported
                if ``by_masking`` is set to :data:`True`.
            replace: Whether to sample with or without replacement.

                Cannot be used with ``by_masking`` set to :data:`True`.
            preserve: Whether to preserve :attr:`object_bounds`, otherwise
                it is discarded.

                Object bounds are re-generated by sorting the randomly generated samples,
                which takes additional time.

                Setting this to :data:`True` has no effect if :attr:`object_bounds`
                is :data:`None`.

                Cannot be used with ``by_masking`` set to :data:`True`.
            by_masking: Whether to sample by masking the primitives.
                If :data:`True`, then the :attr:`mask` attribute set (ignoring any existing mask).
            sample_objects: Whether to sample by objects, i.e., sampling whole objects at once.
                The value of ``size`` is interpreted as the number of objects to sample,
                or the fill factor of objects to sample.
            key: The :func:`jax.random.key` to be used.

        Returns:
            A new random mesh.

        Raises:
            TypeError: If ``by_masking`` is :data:`False` and ``size`` is not an integer.
            ValueError: If ``by_masking`` is :data:`True` and ``replace`` or ``preserve`` are set to :data:`True`.
            ValueError: If ``sample_objects`` is :data:`True` and :attr:`object_bounds` is :data:`None`.
        """
        if sample_objects:
            if self.object_bounds is None:
                msg = "Cannot sample by objects when 'object_bounds' is None."
                raise ValueError(msg)
            num = self.object_bounds.shape[0]
        else:
            num = self.num_primitives
        if by_masking:
            if replace:
                msg = "Cannot sample with replacement when 'by_masking' is True."
                raise ValueError(msg)
            if preserve and not sample_objects:
                msg = "Cannot preserve 'object_bounds' when 'by_masking' is True and 'sample_objects' is False."
                raise ValueError(msg)
            if isinstance(size, int):
                mask = jnp.zeros(num, dtype=bool)
                mask = mask.at[:size].set(True)
                mask = jax.random.permutation(key, mask)
            else:
                mask = jax.random.uniform(key, (num,)) <= size

            if sample_objects:
                indices = jnp.arange(self.num_triangles)
                indices_inside_bounds = (
                    self.object_bounds[:, 0] <= indices[:, None]  # type: ignore[ty:not-subscriptable]
                ) & (indices[:, None] < self.object_bounds[:, 1])  # type: ignore[ty:not-subscriptable]
                triangle_indices = indices_inside_bounds.argmax(
                    axis=1
                )  # Exactly one bound should be true for each triangle
                mask = jnp.take(
                    mask,
                    triangle_indices,
                    indices_are_sorted=True,  # Object bounds are sorted
                )
            elif self.assume_quads:
                mask = jnp.repeat(mask, 2)

            return eqx.tree_at(
                lambda m: (m.object_bounds, m.mask),
                self,
                (self.object_bounds if preserve else None, mask),
                is_leaf=lambda x: x is None,
            )

        if isinstance(size, int):
            indices = jax.random.choice(
                key,
                num,
                shape=(size,),
                replace=replace,
            )
        else:
            msg = "'size' must be an integer when 'by_masking' is False."
            raise TypeError(msg)

        if sample_objects:
            indices = jnp.concatenate([
                jnp.arange(self.object_bounds[i, 0], self.object_bounds[i, 1])  # type: ignore[ty:not-subscriptable]
                for i in indices
            ])
        if self.assume_quads:
            indices *= 2

        if preserve and self.object_bounds is not None:
            indices = jnp.sort(indices)
            object_bounds = jnp.stack(
                (
                    jnp.searchsorted(indices, self.object_bounds[:, 0]),
                    jnp.searchsorted(indices, self.object_bounds[:, 1]),
                ),
                axis=-1,
            )
            # Some object may not have any triangles left after sampling,
            # so we remove them.
            object_bounds = object_bounds.compress(
                object_bounds[:, 0] < object_bounds[:, 1], axis=0
            )
        else:
            object_bounds = None

        if self.assume_quads:
            indices = jnp.stack((indices, indices + 1), axis=-1).reshape(-1)

        return eqx.tree_at(
            lambda m: (
                m.vertices,
                m.triangles,
                m.face_colors,
                m.face_materials,
                m.object_bounds,
                m.mask,
            ),
            self,
            (
                self.vertices,
                self.triangles[indices, :],
                self.face_colors[indices, :] if self.face_colors is not None else None,
                self.face_materials[indices]
                if self.face_materials is not None
                else None,
                object_bounds,
                self.mask[indices] if self.mask is not None else None,
            ),
            is_leaf=lambda x: x is None,
        )

    @overload
    def shuffle(
        self,
        preserve: bool = ...,
        *,
        return_indices: Literal[True],
        key: PRNGKeyArray,
    ) -> tuple[Self, Int[ArrayLike, " num_triangles"]]: ...

    @overload
    def shuffle(
        self,
        preserve: bool = ...,
        *,
        return_indices: Literal[False] = ...,
        key: PRNGKeyArray,
    ) -> Self: ...

    @eqx.filter_jit
    def shuffle(
        self,
        preserve: bool = False,
        *,
        return_indices: bool = False,
        key: PRNGKeyArray,
    ) -> Self | tuple[Self, Int[ArrayLike, " num_triangles"]]:
        """
        Generate a new mesh by randomly shuffling primitives from this geometry.

        Args:
            preserve: Whether to preserve :attr:`object_bounds`, otherwise
                it is discarded.

                .. warning::

                    Not implemented yet.

                Setting this to :data:`True` has no effect if :attr:`object_bounds`
                is :data:`None`.
            return_indices: Whether to return the indices used for shuffling.
            key: The :func:`jax.random.key` to be used.

        Returns:
            A new random mesh.

        Raises:
            NotImplementedError: If `preserve` is :data:`True`.
        """
        if preserve:
            msg = "Preserving object bounds is not implemented yet."
            raise NotImplementedError(msg)

        indices = jax.random.permutation(key, jnp.arange(self.num_primitives))

        if self.assume_quads:
            indices *= 2
            indices = jnp.stack((indices, indices + 1), axis=-1).reshape(-1)

        object_bounds = None

        mesh = eqx.tree_at(
            lambda m: (
                m.vertices,
                m.triangles,
                m.face_colors,
                m.face_materials,
                m.object_bounds,
                m.mask,
            ),
            self,
            (
                self.vertices,
                self.triangles[indices, :],
                self.face_colors[indices, :] if self.face_colors is not None else None,
                self.face_materials[indices]
                if self.face_materials is not None
                else None,
                object_bounds,
                self.mask[indices] if self.mask is not None else None,
            ),
            is_leaf=lambda x: x is None,
        )

        if return_indices:
            return mesh, indices
        return mesh

    def _keep_within(
        self,
        x_min: Float[ArrayLike, ""] | None = None,
        x_max: Float[ArrayLike, ""] | None = None,
        y_min: Float[ArrayLike, ""] | None = None,
        y_max: Float[ArrayLike, ""] | None = None,
        z_min: Float[ArrayLike, ""] | None = None,
        z_max: Float[ArrayLike, ""] | None = None,
        preserve_objects: bool = True,
        keep_any: bool = False,
        clip: bool = False,
    ) -> Self:
        triangle_vertices = self.triangle_vertices
        xs, ys, zs = jnp.unstack(triangle_vertices, axis=-1)

        vertex_mask = jnp.ones_like(xs, dtype=bool)
        if x_min is not None:
            vertex_mask &= xs >= x_min
        if x_max is not None:
            vertex_mask &= xs <= x_max
        if y_min is not None:
            vertex_mask &= ys >= y_min
        if y_max is not None:
            vertex_mask &= ys <= y_max
        if z_min is not None:
            vertex_mask &= zs >= z_min
        if z_max is not None:
            vertex_mask &= zs <= z_max

        mask = (
            jnp.any(vertex_mask, axis=-1) if keep_any else jnp.all(vertex_mask, axis=-1)
        )

        if self.mask is not None:
            mask &= self.mask

        if preserve_objects and self.object_bounds is not None:
            object_ends = self.object_bounds[:, 1]
            object_ids = jnp.searchsorted(
                object_ends, jnp.arange(self.num_triangles), side="right"
            )
            active_mask = (
                self.mask
                if self.mask is not None
                else jnp.ones(self.num_triangles, dtype=bool)
            )
            object_active_counts = (
                jnp
                .zeros(self.object_bounds.shape[0], dtype=int)
                .at[object_ids]
                .add(active_mask.astype(int))
            )
            object_kept_counts = (
                jnp
                .zeros(self.object_bounds.shape[0], dtype=int)
                .at[object_ids]
                .add(mask.astype(int))
            )
            object_mask = (
                object_kept_counts > 0
                if keep_any
                else (object_active_counts > 0)
                & (object_kept_counts == object_active_counts)
            )
            mask = object_mask[object_ids]
            if self.mask is not None:
                mask &= self.mask

        mesh = eqx.tree_at(lambda m: m.mask, self, mask, is_leaf=lambda x: x is None)
        if clip:
            mesh = mesh.clip(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                z_min=z_min,
                z_max=z_max,
            )
        return mesh

    def keep_all_within(
        self,
        x_min: Float[ArrayLike, ""] | None = None,
        x_max: Float[ArrayLike, ""] | None = None,
        y_min: Float[ArrayLike, ""] | None = None,
        y_max: Float[ArrayLike, ""] | None = None,
        z_min: Float[ArrayLike, ""] | None = None,
        z_max: Float[ArrayLike, ""] | None = None,
        *,
        preserve_objects: bool = True,
        clip: bool = False,
    ) -> Self:
        """
        Return a new mesh, keeping only the triangles with all vertices within the given bounds.

        If ``preserve_objects`` is set to :data:`True`, and :attr:`object_bounds` is not
        :data:`None`, then all triangles belonging to the same object as a triangle with all
        vertices within the given bounds are kept, but only if all triangles in that object
        satisfy the bounds.

        Args:
            x_min: The minimum x coordinate.
            x_max: The maximum x coordinate.
            y_min: The minimum y coordinate.
            y_max: The maximum y coordinate.
            z_min: The minimum z coordinate.
            z_max: The maximum z coordinate.
            preserve_objects: Whether to preserve objects.
            clip: Whether to clip the vertices of the returned mesh to the given bounds.

        Returns:
            A new mesh with the triangles filtered according to the given bounds.

        :seealso::

            :meth:`keep_any_within`

        Examples:
            The following example shows how to filter the simple street canyon scene.

            .. plotly::
                :context: reset

                >>> from differt.scene import (
                ...     TriangleScene,
                ...     download_sionna_scenes,
                ...     get_sionna_scene,
                ... )
                >>>
                >>> download_sionna_scenes()  # doctest: +SKIP
                >>> file = get_sionna_scene("simple_street_canyon")
                >>> mesh = TriangleScene.load_xml(file).mesh
                >>> fig = mesh.plot(backend="plotly")
                >>> fig  # doctest: +SKIP

            Here, we keep the objects that have all vertices inside the selected range.

            .. plotly::
                :context:

                >>> fig = mesh.keep_all_within(y_min=-20.0).plot(
                ...     backend="plotly",
                ... )
                >>> fig  # doctest: +SKIP

            By default, and if :attr:`object_bounds` is not :data:`None`, the filtering is done by objects. You can disable this behavior by setting ``preserve_objects`` to :data:`False`.

            .. plotly::
                :context:

                >>> fig = mesh.keep_all_within(
                ...     y_min=-20.0,
                ...     preserve_objects=False,
                ... ).plot(backend="plotly")
                >>> fig  # doctest: +SKIP
        """
        return self._keep_within(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
            preserve_objects=preserve_objects,
            keep_any=False,
            clip=clip,
        )

    def keep_any_within(
        self,
        x_min: Float[ArrayLike, ""] | None = None,
        x_max: Float[ArrayLike, ""] | None = None,
        y_min: Float[ArrayLike, ""] | None = None,
        y_max: Float[ArrayLike, ""] | None = None,
        z_min: Float[ArrayLike, ""] | None = None,
        z_max: Float[ArrayLike, ""] | None = None,
        *,
        preserve_objects: bool = True,
        clip: bool = False,
    ) -> Self:
        """
        Return a new mesh, keeping only the triangles with at least one vertex within the given bounds.

        If ``preserve_objects`` is set to :data:`True`, and :attr:`object_bounds` is not
        :data:`None`, then all triangles belonging to the same object as a triangle with at
        least one vertex within the given bounds are kept.

        Args:
            x_min: The minimum x coordinate.
            x_max: The maximum x coordinate.
            y_min: The minimum y coordinate.
            y_max: The maximum y coordinate.
            z_min: The minimum z coordinate.
            z_max: The maximum z coordinate.
            preserve_objects: Whether to preserve objects.
            clip: Whether to clip the vertices of the returned mesh to the given bounds.

        Returns:
            A new mesh with the triangles filtered according to the given bounds.

        :seealso::

            :meth:`keep_all_within`

        Examples:
            The following example shows how to filter the simple street canyon scene.

            .. plotly::
                :context: reset

                >>> from differt.scene import (
                ...     TriangleScene,
                ...     download_sionna_scenes,
                ...     get_sionna_scene,
                ... )
                >>>
                >>> download_sionna_scenes()  # doctest: +SKIP
                >>> file = get_sionna_scene("simple_street_canyon")
                >>> mesh = TriangleScene.load_xml(file).mesh
                >>> fig = mesh.plot(backend="plotly")
                >>> fig  # doctest: +SKIP

            Here, we keep the objects that have all vertices inside the selected range.

            .. plotly::
                :context:

                >>> fig = mesh.keep_any_within(x_max=0.0).plot(
                ...     backend="plotly",
                ... )
                >>> fig  # doctest: +SKIP

            By default, and if :attr:`object_bounds` is not :data:`None`, the filtering is done by objects. You can disable this behavior by setting ``preserve_objects`` to :data:`False`.

            .. plotly::
                :context:

                >>> fig = mesh.keep_any_within(
                ...     x_max=0.0,
                ...     preserve_objects=False,
                ... ).plot(backend="plotly")
                >>> fig  # doctest: +SKIP

            Finally, we can also clip the vertices of the returned mesh to the given bounds. This is especially useful to trim the plane of the ground in outdoor scenes.

            .. plotly::
                :context:

                >>> fig = mesh.keep_any_within(
                ...     x_max=+15.0,
                ...     clip=True,
                ... ).plot(backend="plotly")
                >>> fig  # doctest: +SKIP
        """
        return self._keep_within(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
            preserve_objects=preserve_objects,
            keep_any=True,
            clip=clip,
        )

    def center(self) -> Self:
        """
        Return a new mesh, centered around the origin.

        The center of the mesh is computed as the center of its bounding box,
        and the mesh is translated so that this center is at the origin in the (x, y) plane.
        The z-coordinate remains unchanged.

        This method is useful for centering scenes after filtering operations that may
        have removed parts of the mesh (e.g., using :meth:`keep_all_within` or :meth:`keep_any_within`).

        Returns:
            A new centered mesh with the same structure but with vertices translated.

        Examples:
            The following example shows how to center a mesh around the origin.

            .. plotly::
                :context: reset

                >>> from differt.geometry import TriangleMesh
                >>>
                >>> mesh = TriangleMesh.box(
                ...     length=4.0, width=2.0, height=1.0
                ... ).translate(jnp.array([2.0, 1.0, 0.0]))
                >>> fig = mesh.plot(backend="plotly")
                >>> fig  # doctest: +SKIP

            After centering, the mesh is translated to the origin in the (x, y) plane:

            .. plotly::
                :context:

                >>> centered = mesh.center()
                >>> fig = centered.plot(backend="plotly")
                >>> fig  # doctest: +SKIP
        """
        (x_min, y_min, _), (x_max, y_max, _) = self.bounding_box
        center = jnp.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5, 0.0])
        return eqx.tree_at(lambda m: m.vertices, self, self.vertices - center)

    def add_ground(
        self,
        x_scale: Float[ArrayLike, ""] = 1.0,
        y_scale: Float[ArrayLike, ""] = 1.0,
        z: Float[ArrayLike, ""] | None = None,
        material_name: str = "itu_concrete",
        face_color: Float[ArrayLike, "3"] = jnp.array([0.539, 0.539, 0.539]),
    ) -> Self:
        """
        Add a ground plane to this mesh.

        The ground plane is a quadrilateral (represented as two triangles) added at the bottom of the mesh,
        i.e., below all existing triangles. The ground plane is scaled based on the bounding box of the mesh
        to create a realistic proportional ground.

        This method is particularly useful for scenes that have been filtered or clipped (e.g., using
        :meth:`keep_all_within` or :meth:`keep_any_within`), as these operations may remove the original ground plane.

        Args:
            x_scale: The scale factor of the ground plane along the x-axis. A value of 1.0 means the ground
                has the same width as the mesh's bounding box. Default is 1.0.
            y_scale: The scale factor of the ground plane along the y-axis. A value of 1.0 means the ground
                has the same depth as the mesh's bounding box. Default is 1.0.
            z: The z coordinate of the ground plane. If :data:`None`, the ground is placed at the minimum
                z coordinate of the mesh's bounding box. Default is :data:`None`.
            material_name: The name of the material for the ground plane. Default is ``"itu_concrete"``.
            face_color: The RGB color of the ground plane faces, with values in the range [0, 1].
                Default is ``[0.539, 0.539, 0.539]`` (concrete gray).

        Returns:
            A new mesh with a ground plane appended.

        Examples:
            The following example shows how to add a ground plane to a simple mesh.

            .. plotly::
                :context: reset

                >>> from differt.geometry import TriangleMesh
                >>>
                >>> mesh = TriangleMesh.box(length=2.0, width=2.0, height=1.0)
                >>> fig = mesh.plot(backend="plotly")
                >>> fig  # doctest: +SKIP

            After adding a ground plane, the mesh includes an additional quadrilateral at the bottom:

            .. plotly::
                :context:

                >>> with_ground = mesh.add_ground()
                >>> fig = with_ground.plot(backend="plotly")
                >>> fig  # doctest: +SKIP

            You can customize the ground plane size and appearance. For example, a larger ground plane with a different color:

            .. plotly::
                :context:

                >>> custom_ground = mesh.add_ground(
                ...     x_scale=2.0, y_scale=2.0, face_color=jnp.array([0.2, 0.2, 0.2])
                ... )
                >>> fig = custom_ground.plot(backend="plotly")
                >>> fig  # doctest: +SKIP
        """
        (x_min, y_min, z_min), (x_max, y_max, _) = self.bounding_box
        if z is None:
            z = z_min

        center = jnp.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5, z])
        u = jnp.array([(x_max - x_min) * x_scale * 0.5, 0.0, 0.0])
        v = jnp.array([0.0, (y_max - y_min) * y_scale * 0.5, 0.0])
        vertices = jnp.array([
            center + u + v,
            center - u + v,
            center - u - v,
            center + u - v,
        ])
        triangles = jnp.array([[0, 1, 2], [0, 2, 3]], dtype=int)

        ground = TriangleMesh(
            vertices=vertices,
            triangles=triangles,
            object_bounds=jnp.array([[0, 2]]),
            face_colors=jnp.array([face_color, face_color]),
            face_materials=jnp.array([0, 0]),
            material_names=(material_name,),
            assume_quads=True,
        )
        return self.append(ground)

    @eqx.filter_jit
    def ray_intersect_any_triangle(
        self,
        ray_origins: Float[Array, "*#batch 3"],
        ray_directions: Float[Array, "*#batch 3"],
        *,
        hit_tol: Float[ArrayLike, ""] | None = None,
    ) -> Bool[Array, "*batch"]:
        """
        Return whether rays intersect any triangle in the mesh.

        Unlike :func:`differt.rt.ray_intersect_any_triangle`, this method is optimized for :class:`TriangleMesh`
        objects when smoothing is disabled and uses :func:`warp.mesh_query_ray_anyhit<warp._src.lang.mesh_query_ray_anyhit>` to accelerate the ray tracing.

        .. warning::

            This method is Warp-accelerated and only supports CPU and CUDA-enabled GPU platforms.
            It does not support TPUs or other non-CUDA GPUs.
            See :doc:`/limitations` for more details.

        Args:
            ray_origins: Origin vertex.
            ray_directions: Ray direction.
            hit_tol: The tolerance applied to check if a ray hits another object or not,
                before it reaches the expected position, i.e., the 'interaction' object.

                Using a non-zero tolerance is required as it would otherwise trigger
                false positives.

                If not specified, the default is one hundred times the epsilon value
                of the currently used floating point dtype.

        Returns:
            A boolean array indicating whether each ray intersects with any triangle in the mesh.
        """
        if self.triangles.shape[0] == 0:
            batch = jnp.broadcast_shapes(
                ray_origins.shape[:-1], ray_directions.shape[:-1]
            )
            return jax.lax.stop_gradient(jnp.full(batch, fill_value=False))

        ray_origins, ray_directions = jnp.broadcast_arrays(ray_origins, ray_directions)

        if hit_tol is None:
            dtype = jnp.result_type(ray_origins, ray_directions, self.vertices)
            hit_tol = 100.0 * jnp.finfo(dtype).eps

        ray_directions, max_t = normalize(ray_directions)
        # NOTE: here, we slightly offset the ray origin so it is not exactly on a face
        # since that can create self-intersections
        offset = hit_tol * max_t
        ray_origins += ray_directions * offset[..., None]
        max_t *= 1 - 2 * hit_tol

        flat_ray_origins = ray_origins.reshape(-1, 3)
        flat_ray_directions = ray_directions.reshape(-1, 3)
        flat_max_t = max_t.reshape(-1)

        triangles = self.triangles
        if self.mask is not None:
            triangles = jnp.where(self.mask[:, None], self.triangles, 0)

        mesh_id = np.uint64(id(self))

        output = jax.lax.platform_dependent(
            jax.lax.stop_gradient(self.vertices),
            triangles,
            jax.lax.stop_gradient(flat_ray_origins),
            jax.lax.stop_gradient(flat_ray_directions),
            jax.lax.stop_gradient(flat_max_t),
            cpu=partial(_ray_intersect_any_triangle_cpu_impl, mesh_id),
            cuda=partial(_ray_intersect_any_triangle_cuda_impl, mesh_id),
        )
        batch = ray_origins.shape[:-1]
        return jax.lax.stop_gradient(output.reshape(batch))

    @eqx.filter_jit
    def first_triangle_hit_by_ray(
        self,
        ray_origins: Float[Array, "*#batch 3"],
        ray_directions: Float[Array, "*#batch 3"],
    ) -> tuple[Int[Array, "*batch"], Float[Array, "*batch"]]:
        """
        Return for each ray, which triangle it intersects first, and if it intersects at all.

        Unlike :func:`differt.rt.first_triangle_hit_by_ray`, this method is optimized for :class:`TriangleMesh`
        objects when smoothing is disabled and uses :func:`warp.mesh_query_ray<warp._src.lang.mesh_query_ray>` to accelerate the ray tracing.

        .. warning::

            This method is Warp-accelerated and only supports CPU and CUDA-enabled GPU platforms.
            It does not support TPUs or other non-CUDA GPUs.
            See :doc:`/limitations` for more details.

        Args:
            ray_origins: Origin vertex.
            ray_directions: Ray direction.

        Returns:
            For each ray, return the index and to distance to the first triangle hit.

            If no triangle is hit, the index is set to ``-1`` and
            the distance is set to :data:`inf<numpy.inf>`.

        .. note::

            The returned distance is fully differentiable with respect to the ray origins,
            ray directions, and mesh vertices, using JAX's automatic differentiation.
        """
        if self.triangles.shape[0] == 0:
            batch = jnp.broadcast_shapes(
                ray_origins.shape[:-1], ray_directions.shape[:-1]
            )
            return (
                jax.lax.stop_gradient(jnp.full(batch, -1, dtype=int)),
                jnp.full(batch, jnp.inf, dtype=float),
            )

        ray_origins, ray_directions = jnp.broadcast_arrays(ray_origins, ray_directions)

        flat_ray_origins = ray_origins.reshape(-1, 3)
        flat_ray_directions = ray_directions.reshape(-1, 3)

        triangles = self.triangles
        if self.mask is not None:
            triangles = jnp.where(self.mask[:, None], self.triangles, 0)

        mesh_id = np.uint64(id(self))

        out_faces, out_t = _first_triangle_hit_by_ray_helper(
            mesh_id,
            self.vertices,
            triangles,
            flat_ray_origins,
            flat_ray_directions,
        )

        batch = ray_origins.shape[:-1]

        return (
            jax.lax.stop_gradient(out_faces.reshape(batch)),
            out_t.reshape(batch),
        )

    @eqx.filter_jit
    def triangles_visible_from_vertex(
        self,
        vertex: Float[Array, "*#batch 3"],
        num_rays: int = int(1e6),
    ) -> Bool[Array, "*batch num_triangles"]:
        """
        Return whether triangles are visible from vertex positions.

        Unlike :func:`differt.rt.triangles_visible_from_vertex`, this method is optimized for :class:`TriangleMesh`
        objects when smoothing is disabled and uses :func:`warp.mesh_query_ray<warp._src.lang.mesh_query_ray>` to accelerate the ray tracing.

        .. warning::

            This method is Warp-accelerated and only supports CPU and CUDA-enabled GPU platforms.
            It does not support TPUs or other non-CUDA GPUs.
            See :doc:`/limitations` for more details.

        Args:
            vertex: Vertex, used as origin of the rays.
            num_rays: The number of rays to launch.

        Returns:
            A boolean array indicating whether each triangle is visible from each vertex.
        """
        if self.triangles.shape[0] == 0:
            return jax.lax.stop_gradient(
                jnp.zeros((*vertex.shape[:-1], self.num_triangles), dtype=bool)
            )

        triangle_vertices = self.triangle_vertices
        triangle_centers = triangle_vertices.mean(axis=-2, keepdims=True)
        world_vertices = jnp.concat(
            (triangle_vertices, triangle_centers), axis=-2
        ).reshape(*triangle_vertices.shape[:-3], -1, 3)

        active_triangles = self.mask
        if active_triangles is not None:
            active_vertices = jnp.repeat(active_triangles, 4, axis=-1)
        else:
            active_vertices = None

        # [*batch 3]
        ray_origins = vertex

        # [*batch 2 3]
        frustum = viewing_frustum(
            ray_origins,
            world_vertices,
            active_vertices=active_vertices,
        )

        # [*batch num_rays 3]
        ray_directions = jnp.vectorize(
            lambda n, frustum: fibonacci_lattice(n, frustum=frustum),
            excluded={0},
            signature="(2,3)->(n,3)",
        )(num_rays, frustum)

        ray_origins, ray_directions = jnp.broadcast_arrays(
            ray_origins[..., None, :], ray_directions
        )

        flat_ray_origins = ray_origins.reshape(-1, 3)
        flat_ray_directions = ray_directions.reshape(-1, 3)

        total_batches = flat_ray_origins.shape[0] // num_rays
        num_triangles = self.num_triangles

        triangles = self.triangles
        if self.mask is not None:
            triangles = jnp.where(self.mask[:, None], self.triangles, 0)

        mesh_id = np.uint64(id(self))

        out_visible = jax.lax.platform_dependent(
            jax.lax.stop_gradient(self.vertices),
            triangles,
            jax.lax.stop_gradient(flat_ray_origins),
            jax.lax.stop_gradient(flat_ray_directions),
            cpu=partial(
                _triangles_visible_from_vertex_cpu_impl,
                mesh_id,
                num_rays,
                num_triangles,
                total_batches,
            ),
            cuda=partial(
                _triangles_visible_from_vertex_cuda_impl,
                mesh_id,
                num_rays,
                num_triangles,
                total_batches,
            ),
        )

        batch_shape = vertex.shape[:-1]
        return jax.lax.stop_gradient(out_visible.reshape(*batch_shape, num_triangles))
