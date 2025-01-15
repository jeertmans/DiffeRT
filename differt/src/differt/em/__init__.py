"""Electromagnetic (EM) fields utilities."""

__all__ = (
    "Antenna",
    "BaseAntenna",
    "Dipole",
    "F",
    "HWDipolePattern",
    "InteractionType",
    "L_i",
    "Material",
    "RadiationPattern",
    "ShortDipole",
    "ShortDipolePattern",
    "c",
    "diffraction_coefficients",
    "epsilon_0",
    "fresnel_coefficients",
    "fspl",
    "lengths_to_delays",
    "materials",
    "mu_0",
    "path_delays",
    "pointing_vector",
    "reflection_coefficients",
    "refraction_coefficients",
    "refractive_indices",
    "sp_directions",
    "sp_rotation_matrix",
    "transition_matrices",
    "z_0",
)

from ._antenna import (
    Antenna,
    BaseAntenna,
    Dipole,
    HWDipolePattern,
    RadiationPattern,
    ShortDipole,
    ShortDipolePattern,
    pointing_vector,
)
from ._constants import c, epsilon_0, mu_0, z_0
from ._fresnel import (
    fresnel_coefficients,
    reflection_coefficients,
    refraction_coefficients,
    refractive_indices,
)
from ._interaction_type import InteractionType
from ._material import Material, materials
from ._utd import F, L_i, diffraction_coefficients
from ._utils import (
    fspl,
    lengths_to_delays,
    path_delays,
    sp_directions,
    sp_rotation_matrix,
    transition_matrices,
)
