from enum import IntEnum, unique


@unique
class InteractionType(IntEnum):
    """Enumeration of interaction types."""

    NONE = -1
    """No interaction (placeholder)."""
    REFLECTION = 0
    """Specular reflection on a surface."""
    DIFFRACTION = 1
    """Diffraction on an edge."""
    SCATTERING = 2
    """Scattering on a surface."""
