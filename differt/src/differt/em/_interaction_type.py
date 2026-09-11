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
    """Scattering on a rough surface."""
    TRANSMISSION = 3
    """Transmission through a surface."""
    RIS = 4
    """Interaction with a Reconfigurable Intelligent Surface (RIS)."""


SpecularReflection = InteractionType.REFLECTION
"""Alias for :attr:`InteractionType.REFLECTION`."""
Diffraction = InteractionType.DIFFRACTION
"""Alias for :attr:`InteractionType.DIFFRACTION`."""
Scattering = InteractionType.SCATTERING
"""Alias for :attr:`InteractionType.SCATTERING`."""
Transmission = InteractionType.TRANSMISSION
"""Alias for :attr:`InteractionType.TRANSMISSION`."""
RIS = InteractionType.RIS
"""Alias for :attr:`InteractionType.RIS`."""
