"""Mesh geometry made of triangles and utilities."""
from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from chex import dataclass
from jaxtyping import Array, Bool, Float, UInt

from .. import _core
from ..plotting import draw_mesh
from .utils import normalize


def triangles_contain_vertices_assuming_inside_same_plane(
    triangle_vertices: Float[Array, "*batch 3 3"], vertices: Float[Array, " *batch 3"]
) -> Bool[Array, " *batch"]:
    """
    Return whether each triangle contains the corresponding vertex, but
    assuming the vertex lies in the same plane as the triangle.

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


def paths_intersect_triangles(
    paths: Float[Array, "*batch path_length 3"],
    triangle_vertices: Float[Array, "num_triangles 3 3"],
) -> Bool[Array, " *batch"]:
    """
    Return whether each path intersect with any of the triangles.

    Args:
        paths: An array of ray paths of the same length.
        triangle_vertices: An array of triangle vertices.

    Returns:
        A boolean array indicating whether vertices are in the corresponding triangles or not.
    """
    pass


@dataclass
class TriangleMesh:
    """
    A simple geometry made of triangles.
    """

    _mesh: _core.geometry.triangle_mesh.TriangleMesh

    @cached_property
    def triangles(self) -> UInt[Array, "num_triangles 3"]:
        """The triangle indices."""
        return jnp.asarray(self._mesh.triangles, dtype=jnp.uint32)

    @cached_property
    def vertices(self) -> Float[Array, "num_vertices 3"]:
        """The vertices."""
        return jnp.asarray(self._mesh.vertices)

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
    def load_obj(cls, file: Path) -> TriangleMesh:
        """
        Load a triangle mesh from a Wavefront .obj file.

        Currently, only vertices and triangles are loaded. Triangle normals
        are computed afterward (when first accessed).

        This method will fail if it contains any geometry that is not a triangle.

        Args:
            file: The path to the wavefront .obj file.

        Returns:
            The corresponding mesh containing only triangles.
        """
        return cls(_mesh=_core.geometry.triangle_mesh.TriangleMesh.load_obj(str(file)))

    def plot(self, *args: Any, **kwargs: Any) -> None:
        """*TODO*."""
        return draw_mesh(
            vertices=np.asarray(self.vertices),
            faces=np.asarray(self.triangles),
            *args,
            **kwargs,
        )
