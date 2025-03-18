from differt_core.geometry import TriangleMesh

class TriangleScene:
    mesh: list[TriangleMesh]

    @classmethod
    def load_xml(cls, file: str) -> TriangleScene: ...
