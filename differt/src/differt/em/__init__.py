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
    "compute_cir",
    "compute_received_power",
    "compute_transition_matrices",
    "diffraction_coefficients",
    "epsilon_0",
    "fresnel_coefficients",
    "fspl",
    "length_to_delay",
    "materials",
    "mu_0",
    "path_delay",
    "poynting_vector",
    "reflection_coefficients",
    "refraction_coefficients",
    "refractive_index",
    "sp_directions",
    "sp_rotation_matrix",
    "transition_matrix",
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
    poynting_vector,
)
from ._cir import compute_cir, compute_received_power, compute_transition_matrices
from ._constants import c, epsilon_0, mu_0, z_0
from ._fresnel import (
    fresnel_coefficients,
    reflection_coefficients,
    refraction_coefficients,
    refractive_index,
)
from ._interaction_type import InteractionType
from ._material import Material, materials
from ._utd import F, L_i, diffraction_coefficients
from ._utils import (
    fspl,
    length_to_delay,
    path_delay,
    sp_directions,
    sp_rotation_matrix,
    transition_matrix,
)

