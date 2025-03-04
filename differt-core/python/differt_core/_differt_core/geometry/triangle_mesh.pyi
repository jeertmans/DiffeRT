# pyright: reportMissingTypeArgument=false
import numpy as np
from jaxtyping import Float, Int, UInt

class TriangleMesh:
    vertices: Float[np.ndarray, "num_vertices 3"]
    triangles: UInt[np.ndarray, "num_triangles 3"]
    face_colors: Float[np.ndarray, "num_triangles 3"] | None
    face_materials: Int[np.ndarray, " num_triangles"] | None
    material_names: list[str]
    object_bounds: UInt[np.ndarray, "num_objects 2"] | None

    def append(self, other: TriangleMesh) -> None: ...
    @classmethod
    def load_obj(cls, file: str) -> TriangleMesh: ...
    @classmethod
    def load_ply(cls, file: str) -> TriangleMesh: ...
