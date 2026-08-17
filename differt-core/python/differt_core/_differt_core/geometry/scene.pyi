from os import PathLike

from differt_core.geometry import TriangleMesh

class Scene:
    mesh: list[TriangleMesh]

    @classmethod
    def load_xml(cls, file: str | PathLike[str]) -> Scene: ...
