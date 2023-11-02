from __future__ import annotations

from functools import cached_property
from pathlib import Path

import jax.numpy as jnp
import open3d as o3d
import plotly.graph_objects as go
from chex import dataclass
from jaxtyping import Array, Float, UInt

from .utils import pairwise_cross


@dataclass
class TriangleMesh:
    mesh: o3d.geometry.TriangleMesh

    @cached_property
    def indices(self) -> UInt[Array, "num_triangles 3"]:
        return jnp.asarray(self.mesh.triangles, dtype=int)

    @cached_property
    def vertices(self) -> Float[Array, "num_vertices 3"]:
        return jnp.asarray(self.mesh.vertices)

    @cached_property
    def normals(self) -> Float[Array, "num_triangles 3"]:
        return jnp.asarray(self.mesh.triangle_normals)

    @cached_property
    def diffraction_edges(self) -> UInt[Array, "num_edges 3"]:
        all_vertices = jnp.take(self.vertices, self.indices)
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

    def plot(self):
        x, y, z = self.vertices.T
        i = self.indices[:, 0]
        j = self.indices[:, 1]
        k = self.indices[:, 2]
        fig = go.Figure(
            data=[
                go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=i,
                    j=j,
                    k=k,
                )
            ]
        )
        fig.show()
