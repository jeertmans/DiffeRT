"""Mesh geometry made of triangles and utilities."""
# ruff: noqa: ERA001

from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype as typechecker
from jaxtyping import Array, ArrayLike, Bool, Float, Int, PRNGKeyArray, jaxtyped

import differt_core.geometry.triangle_mesh
from differt.plotting import draw_mesh
from differt.rt.utils import rays_intersect_triangles

from .utils import normalize, orthogonal_basis, rotation_matrix_along_axis


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def triangles_contain_vertices_assuming_inside_same_plane(
    triangle_vertices: Float[Array, "*batch 3 3"],
    vertices: Float[Array, "*batch 3"],
) -> Bool[Array, " *batch"]:
    """
    Return whether each triangle contains the corresponding vertex, but assuming the vertex lies in the same plane as the triangle.

    This is especially useful when combined with the
    :func:`image_method<differt.rt.image_method.image_method>`, as the paths returned
    will also lie in the same plane as the mirrors, but may be outside of the actual reflector,
    e.g., a triangular surface.

    Args:
        triangle_vertices: an array of triangle vertices.
        vertices: an array of vertices that will be checked.

    Returns:
        A boolean array indicating whether vertices are in the corresponding triangles or not.
    """
    # [*batch 3]
    p0 = triangle_vertices[..., 0, :]
    p1 = triangle_vertices[..., 1, :]
    p2 = triangle_vertices[..., 2, :]

    # Vectors from test vertex to every triangle vertex
    # [*batch 3]
    u0 = p0 - vertices
    u1 = p1 - vertices
    u2 = p2 - vertices

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


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def paths_intersect_triangles(
    paths: Float[Array, "*batch path_length 3"],
    triangle_vertices: Float[Array, "num_triangles 3 3"],
    epsilon: Float[ArrayLike, " "] = 1e-6,
) -> Bool[Array, " *batch"]:
    """
    Return whether each path intersect with any of the triangles.

    Args:
        paths: An array of ray paths of the same length.
        triangle_vertices: An array of triangle vertices.
        epsilon: A small tolerance threshold that excludes
            a small portion of the path, to avoid indicating intersection
            when a path *bounces off* a triangle.

    Returns:
        A boolean array indicating whether vertices are in the corresponding triangles or not.
    """
    ray_origins = paths[..., :-1, :]
    ray_directions = jnp.diff(paths, axis=-2)

    t, hit = rays_intersect_triangles(
        ray_origins,
        ray_directions,
        jnp.broadcast_to(triangle_vertices, (*ray_origins.shape, 3)),
    )
    intersect = (t < (1 - epsilon)) & hit
    return jnp.any(intersect, axis=(0, 2))


@jaxtyped(typechecker=typechecker)
class TriangleMesh(eqx.Module):
    """
    A simple geometry made of triangles.

    Args:
        vertices: The array of triangle vertices.
        triangles: The array of triangle indices.
    """

    vertices: Float[Array, "num_vertices 3"] = eqx.field(converter=jnp.asarray)
    """The array of triangle vertices."""
    triangles: Int[Array, "num_triangles 3"] = eqx.field(converter=jnp.asarray)
    """The array of triangle indices."""
    face_colors: Optional[Float[Array, "num_triangles 3"]] = eqx.field(
        converter=lambda x: jnp.asarray(x) if x is not None else None, default=None
    )
    """The array of face colors.

    The array contains the face colors, as RGB triplets,
    with a special placeholder value of :data:`(-1, -1, -1)`.
    This attribute is :data:`None` if all face colors are unset.
    """
    face_materials: Optional[Int[Array, " num_triangles"]] = eqx.field(
        converter=lambda x: jnp.asarray(x) if x is not None else None, default=None
    )
    """The array of face materials.

    The array contains the material indices,
    with a special placeholder value of :data:`-1`.
    The obtain the name of the material, see :attr:`material_names`.
    This attribute is :data:`None` if all face materials are unset.
    """
    material_names: tuple[str, ...] = eqx.field(converter=tuple, default_factory=tuple)
    """The list of material names."""
    object_bounds: Optional[Int[Array, "num_objects 2"]] = eqx.field(
        converter=lambda x: jnp.asarray(x) if x is not None else None, default=None
    )
    """The array of object indices.

    If the present mesh contains multiple objects, usually as a result of appending
    multiple meshes together, this array contain start end end indices for each sub mesh.
    """

    @classmethod
    def from_core(
        cls, core_mesh: differt_core.geometry.triangle_mesh.TriangleMesh
    ) -> "TriangleMesh":
        """
        Return a triangle mesh from a mesh created by the :mod:`differt_core` module.

        Args:
            core_mesh: The mesh from the core module.

        Returns:
            The corresponding mesh.
        """
        return cls(
            vertices=core_mesh.vertices,
            triangles=core_mesh.triangles.astype(int),
            face_colors=core_mesh.face_colors,
            face_materials=core_mesh.face_materials,
            material_names=tuple(core_mesh.material_names),
            object_bounds=core_mesh.object_bounds.astype(int)
            if core_mesh.object_bounds is not None
            else None,
        )

    @property
    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def normals(self) -> Float[Array, "num_triangles 3"]:
        """The triangle normals."""
        vertices = jnp.take(self.vertices, self.triangles, axis=0)
        vectors = jnp.diff(vertices, axis=1)
        normals = jnp.cross(vectors[:, 0, :], vectors[:, 1, :])

        return normalize(normals)[0]

    @property
    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def diffraction_edges(self) -> Int[Array, "num_edges 3"]:
        """The diffraction edges."""
        raise NotImplementedError

    @property
    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def bounding_box(self) -> Float[Array, "2 3"]:
        """The bounding box (min. and max. coordinates)."""
        return jnp.vstack(
            (jnp.min(self.vertices, axis=0), jnp.max(self.vertices, axis=0)),
        )

    @classmethod
    @eqx.filter_jit
    def empty(cls) -> "TriangleMesh":
        """
        Create an empty mesh.

        Returns:
            A new empty scene.
        """
        return cls(vertices=jnp.empty((0, 3)), triangles=jnp.empty((0, 3), dtype=int))

    @classmethod
    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def plane(
        cls,
        vertex: Float[Array, "3"],
        *other_vertices: Float[Array, "3"],
        normal: Optional[Float[Array, "3"]] = None,
        side_length: Float[ArrayLike, " "] = 1.0,
        rotate: Optional[Float[ArrayLike, " "]] = None,
    ) -> "TriangleMesh":
        """
        Create an plane mesh, made of two triangles.

        Args:
            vertex: The center of the plane.
            other_vertices: Two other vertices that define the plane.

                This or ``normal`` is required.
            normal: The plane normal.

                Must be of unit length.
            side_length: The side length of the plane.
            rotate: An optional rotation angle, in radians,
                to be applied around the normal of the plane
                and its center.

        Returns:
            A new plane mesh.

        Raises:
            ValueError: If one of two ``other_vertices`` or ``normal``
                were not provided.
        """
        if (other_vertices == ()) == (normal is None):
            msg = "You must specify one of `other_vertices` or `normal`, not both."
            raise ValueError(msg)
        if other_vertices:
            if len(other_vertices) != 2:  # noqa: PLR2004
                msg = (
                    "You must provide exactly 3 vertices to create a new plane, "
                    f"but you provided {len(other_vertices) + 1}."
                )
                raise ValueError(msg)
            u = other_vertices[0] - vertex
            v = other_vertices[1] - vertex
            w = jnp.cross(u, v)
            (normal, _) = normalize(w)

        u, v = orthogonal_basis(normal, normalize=True)

        s = 0.5 * side_length

        vertices = s * jnp.array([u + v, v - u, -u - v, u - v])

        if rotate:
            rotation_matrix = rotation_matrix_along_axis(rotate, normal)
            vertices = (rotation_matrix @ vertices.T).T

        vertices += vertex

        triangles = jnp.array([[0, 1, 2], [0, 2, 3]], dtype=int)
        return cls(vertices=vertices, triangles=triangles)

    @property
    def is_empty(self) -> bool:
        """Whether this scene has no triangle."""
        return self.triangles.size == 0

    @classmethod
    def load_obj(cls, file: str) -> "TriangleMesh":
        """
        Load a triangle mesh from a Wavefront .obj file.

        Currently, only vertices and triangles are loaded. Triangle normals
        are computed afterward (when first accessed).

        Args:
            file: The path to the Wavefront .obj file.

        Returns:
            The corresponding mesh containing only triangles.
        """
        core_mesh = differt_core.geometry.triangle_mesh.TriangleMesh.load_obj(file)
        return cls.from_core(core_mesh)

    @classmethod
    def load_ply(cls, file: str) -> "TriangleMesh":
        """
        Load a triangle mesh from a Stanford PLY .ply file.

        Currently, only vertices and triangles are loaded. Triangle normals
        are computed afterward (when first accessed).

        Args:
            file: The path to the Stanford PLY .ply file.

        Returns:
            The corresponding mesh containing only triangles.
        """
        core_mesh = differt_core.geometry.triangle_mesh.TriangleMesh.load_ply(file)
        return cls.from_core(core_mesh)

    def plot(self, **kwargs: Any) -> Any:
        """
        Plot this mesh on a 3D scene.

        Args:
            kwargs: Keyword arguments passed to
                :py:func:`draw_mesh<differt.plotting.draw_mesh>`.

        Returns:
            The resulting plot output.
        """
        if "face_colors" not in kwargs:
            kwargs["face_colors"] = self.face_colors

        return draw_mesh(
            vertices=np.asarray(self.vertices),
            triangles=np.asarray(self.triangles),
            **kwargs,
        )

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def sample(
        self,
        size: int,
        replace: bool = False,
        *,
        key: PRNGKeyArray,
    ) -> "TriangleMesh":
        """
        Generate a new mesh by randomly sampling triangles from this geometry.

        Args:
            size: The size of the sample, i.e., the number of triangles.
            replace: Whether to sample with or without replacement.
            key: The :func:`jax.random.PRNGKey` to be used.

        Returns:
            A new random mesh.
        """
        triangles = self.triangles[
            jax.random.choice(
                key,
                self.triangles.shape[0],
                shape=(size,),
                replace=replace,
            ),
            :,
        ]
        return TriangleMesh(vertices=self.vertices, triangles=triangles)
