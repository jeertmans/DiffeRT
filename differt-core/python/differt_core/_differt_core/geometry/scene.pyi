from os import PathLike

from differt_core.geometry import Mesh

class Scene:
    mesh: Mesh

    @classmethod
    def load_xml(cls, file: str | PathLike[str]) -> Scene: ...
