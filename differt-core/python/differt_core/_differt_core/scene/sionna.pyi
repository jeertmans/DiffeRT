class SionnaScene:
    shapes: dict[str, Shape]
    materials: dict[str, Material]

    @classmethod
    def load_xml(cls, file: str) -> SionnaScene: ...

class Material:
    name: str
    id: str
    color: tuple[float, float, float]
    thickness: float | None

class Shape:
    type: str
    id: str
    file: str
    material_id: str
