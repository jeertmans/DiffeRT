from differt_core.geometry.triangle_mesh import TriangleMesh

from .sionna import Material

class TriangleScene:
    meshes: list[TriangleMesh]
    materials: list[Material]

    @classmethod
    def load_xml(cls, file: str) -> TriangleScene: ...
