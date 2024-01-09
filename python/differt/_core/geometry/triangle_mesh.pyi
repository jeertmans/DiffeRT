import numpy as np
from jaxtyping import Float, UInt

class TriangleMesh:
    vertices: Float[np.ndarray, "num_vertices 3"]
    triangles: UInt[np.ndarray, "num_triangles 3"]

    @classmethod
    def load_obj(cls, file: str) -> TriangleMesh: ...
