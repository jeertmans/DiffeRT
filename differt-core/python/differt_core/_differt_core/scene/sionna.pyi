class SionnaScene:
    shapes: dict[str, Shape]
    materials: dict[str, Material]

    @classmethod
    def load_xml(cls, file: str) -> SionnaScene: ...

class Material:
    id: str
    rgb: tuple[float, float, float]

class Shape:
    type: str
    id: str
    file: str
    material_id: str
