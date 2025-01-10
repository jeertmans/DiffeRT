from enum import IntEnum, unique


@unique
class InteractionType(IntEnum):
    """Enumeration of interaction types."""

    REFLECTION = 0
    """Specular reflection on an surface."""
    DIFFRACTION = 1
    """Diffraction on an edge."""
    SCATTERING = 2
    """Scattering on a surface."""
