from enum import IntEnum, unique


@unique
class InteractionType(IntEnum):
    """Enumeration of interaction types.

    .. note::

        This enum is also re-exported directly from the top-level :mod:`differt` package
        (e.g., ``from differt import InteractionType``).
    """

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
"""Ergonomic alias for :attr:`InteractionType.REFLECTION`.

Also re-exported directly from the top-level :mod:`differt` package
(e.g., ``from differt import SpecularReflection``).
"""
Diffraction = InteractionType.DIFFRACTION
"""Ergonomic alias for :attr:`InteractionType.DIFFRACTION`.

Also re-exported directly from the top-level :mod:`differt` package
(e.g., ``from differt import Diffraction``).
"""
Scattering = InteractionType.SCATTERING
"""Ergonomic alias for :attr:`InteractionType.SCATTERING`.

Also re-exported directly from the top-level :mod:`differt` package
(e.g., ``from differt import Scattering``).
"""
Transmission = InteractionType.TRANSMISSION
"""Ergonomic alias for :attr:`InteractionType.TRANSMISSION`.

Also re-exported directly from the top-level :mod:`differt` package
(e.g., ``from differt import Transmission``).
"""
RIS = InteractionType.RIS
"""Ergonomic alias for :attr:`InteractionType.RIS`.

Also re-exported directly from the top-level :mod:`differt` package
(e.g., ``from differt import RIS``).
"""
