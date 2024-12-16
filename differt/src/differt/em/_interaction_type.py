from enum import IntEnum, unique


@unique
class InteractionType(IntEnum):
    """Enumeration of interaction types."""

    REFLECTION = 0
    DIFFRACTION = 1
    SCATTERING = 2
