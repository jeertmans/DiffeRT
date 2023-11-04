from __future__ import annotations

from functools import cached_property
from pathlib import Path

import jax.numpy as jnp
import open3d as o3d
import plotly.graph_objects as go
from chex import dataclass
from jaxtyping import Array, Bool, Float, UInt

from .utils import pairwise_cross


def triangles_contain_vertices_assuming_inside_same_plane(
    triangle_vertices: Float[Array, "*batch 3 3"], vertices: Float[Array, "*batch 3"]
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


@dataclass
class TriangleMesh:
    mesh: o3d.geometry.TriangleMesh

    @cached_property
    def triangles(self) -> UInt[Array, "num_triangles 3"]:
        """Return the array of triangle indices."""
        return jnp.asarray(self.mesh.triangles, dtype=int)

    @cached_property
    def vertices(self) -> Float[Array, "num_vertices 3"]:
        """Return the array of vertices."""
        return jnp.asarray(self.mesh.vertices)

    @cached_property
    def normals(self) -> Float[Array, "num_triangles 3"]:
        return jnp.asarray(self.mesh.triangle_normals)

    @cached_property
    def diffraction_edges(self) -> UInt[Array, "num_edges 3"]:
        all_vertices = jnp.take(self.vertices, self.indices, axis=0)
        normals = self.normals

        print(normals)

        cross = pairwise_cross(normals, normals)
        n = jnp.linalg.norm(cross, axis=-1)
        print(n)
        print(n.shape)
        print(cross)
        print(cross.shape)

        print(all_vertices)
        return len(all_vertices)
        return jnp.asarray(self.mesh.get_non_manifold_edges(False))

    @classmethod
    def load_geojson(cls, file: Path, default_height: float = 1.0) -> TriangleMesh:
        pass

    @classmethod
    def load_obj(cls, file: Path) -> TriangleMesh:
        mesh = o3d.io.read_triangle_mesh(str(file)).compute_triangle_normals()
        return cls(mesh=mesh)

    def plot(self, *args, **kwargs):
        x, y, z = self.vertices.T
        i = self.triangles[:, 0]
        j = self.triangles[:, 1]
        k = self.triangles[:, 2]
        fig = go.Figure(
            data=[
                go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=i,
                    j=j,
                    k=k,
                    *args, **kwargs
                )
            ]
        )
        transmitters = jnp.array([0.0, 4.9352, 22.0])
        receivers = jnp.array([0.0, 10.034, 1.5])

        return fig
