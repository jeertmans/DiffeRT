from differt_core.geometry.triangle_mesh import TriangleMesh

class TriangleScene:
    mesh: list[TriangleMesh]

    @classmethod
    def load_xml(cls, file: str) -> TriangleScene: ...
