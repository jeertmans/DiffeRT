from ..geometry.triangle_mesh import TriangleMesh

class TriangleScene:
    mesh: TriangleMesh
    mesh_ids: dict[str, slice]

    @classmethod
    def load_xml(cls, file: str) -> TriangleMesh: ...
