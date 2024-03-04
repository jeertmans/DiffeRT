"""Mesh geometry made of triangles and utilities."""

from functools import cached_property
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, UInt, jaxtyped

from .. import _core
from ..plotting import draw_mesh
from ..rt.utils import rays_intersect_triangles
from .utils import normalize


@jaxtyped(typechecker=typechecker)
def triangles_contain_vertices_assuming_inside_same_plane(
    triangle_vertices: Float[Array, "*batch 3 3"], vertices: Float[Array, "*batch 3"]
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

    Return:
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


@jaxtyped(typechecker=typechecker)
def paths_intersect_triangles(
    paths: Float[Array, "*batch path_length 3"],
    triangle_vertices: Float[Array, "num_triangles 3 3"],
    epsilon: float = 1e-6,
) -> Bool[Array, " *batch"]:
    """
    Return whether each path intersect with any of the triangles.

    Args:
        paths: An array of ray paths of the same length.
        triangle_vertices: An array of triangle vertices.
        epsilon: A small tolerance threshold that excludes
            a small portion of the path, to avoid indicating intersection
            when a path *bounces off* a triangle.

    Return:
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
    triangles: UInt[Array, "num_triangles 3"] = eqx.field(converter=jnp.asarray)
    """The array of triangle indices."""

    @cached_property
    def normals(self) -> Float[Array, "num_triangles 3"]:
        """The triangle normals."""
        vertices = jnp.take(self.vertices, self.triangles, axis=0)
        vectors = jnp.diff(vertices, axis=1)
        normals = jnp.cross(vectors[:, 0, :], vectors[:, 1, :])

        return normalize(normals)[0]

    @cached_property
    def diffraction_edges(self) -> UInt[Array, "num_edges 3"]:
        """The diffraction edges."""
        raise NotImplementedError

    @classmethod
    def load_obj(cls, file: str) -> "TriangleMesh":
        """
        Load a triangle mesh from a Wavefront .obj file.

        Currently, only vertices and triangles are loaded. Triangle normals
        are computed afterward (when first accessed).

        This method will fail if it contains any geometry that is not a triangle.

        Args:
            file: The path to the Wavefront .obj file.

        Return:
            The corresponding mesh containing only triangles.
        """
        mesh = _core.geometry.triangle_mesh.TriangleMesh.load_obj(file)
        return cls(
            vertices=mesh.vertices,
            triangles=mesh.triangles,
        )

    def plot(self, **kwargs: Any) -> Any:
        """
        Plot this mesh on a 3D scene.

        Args:
            kwargs: Keyword arguments passed to
                :py:func:`draw_mesh<differt.plotting.draw_mesh>`.

        Return:
            The resulting plot output.
        """
        return draw_mesh(
            vertices=np.asarray(self.vertices),
            triangles=np.asarray(self.triangles),
            **kwargs,
        )
