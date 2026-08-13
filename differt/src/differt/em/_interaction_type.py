from enum import IntEnum, unique


@unique
class InteractionType(IntEnum):
    """Enumeration of interaction types."""

    REFLECTION = 0
    """Specular reflection on a surface."""
    DIFFRACTION = 1
    """Diffraction on an edge."""
    SCATTERING = 2
    """Scattering on a rough surface."""
    TRANSMISSION = 3
    """Transmission through a surface."""
    RIS = 4
    """Interaction with a Reconfigurable Intelligent Surface (RIS)."""
