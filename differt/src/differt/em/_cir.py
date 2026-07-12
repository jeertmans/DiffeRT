"""Channel Impulse Response (CIR) computation utilities.

This module provides end-to-end electromagnetic field computation
from traced geometric paths to polarimetric Channel Impulse Responses.

The CIR for a given transmitter-receiver pair is:

.. math::
    h(\\tau, t) = \\sum_{p=1}^{P} a_p \\, e^{j 2\\pi f_{D,p} t} \\, \\delta(\\tau - \\tau_p)

where :math:`a_p \\in \\mathbb{C}^{2 \\times 2}` is the polarimetric channel gain
matrix, :math:`\\tau_p` is the propagation delay, and :math:`f_{D,p}` is the
Doppler shift (currently not implemented).

The complex amplitude matrix is computed by chaining transition matrices
along the path:

.. math::
    \\mathbf{E}_{rx} = \\left[ \\prod_{\\ell=L}^{1} \\mathbf{T}^{(\\ell)} e^{-j k d_{\\ell}} \\right] \\mathbf{E}_{tx}
"""

__all__ = (
    "compute_cir",
    "compute_received_power",
    "compute_transition_matrices",
)

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Complex, Float, Int

from differt.em._constants import c
from differt.em._interaction_type import InteractionType
from differt.geometry._utils import normalize


@jax.jit
def compute_transition_matrices(
    vertices: Float[ArrayLike, "*batch path_length 3"],
    normals: Float[ArrayLike, "*batch num_interactions 3"],
    interaction_types: Int[ArrayLike, "*batch num_interactions"],
    relative_permittivities: Complex[ArrayLike, "*batch num_interactions"],
    wavenumber: Float[ArrayLike, ""],
) -> Complex[Array, "*batch 2 2"]:
    r"""Compute the cumulative 2×2 transition matrix for a multipath trajectory.

    For each interaction along the path, a local transition matrix is constructed
    from the Fresnel reflection/refraction coefficients (for reflections/transmissions)
    or the UTD diffraction coefficients (for diffractions). These are chained
    together via matrix multiplication in reverse order.

    Args:
        vertices: Path vertices, starting from TX and ending at RX.
            Shape ``[*batch, path_length, 3]`` where ``path_length = num_interactions + 2``.
        normals: Surface normals at each interaction point.
        interaction_types: Type of each interaction (see :class:`InteractionType`).
        relative_permittivities: Complex relative permittivity of each interacting surface.
        wavenumber: The wavenumber :math:`k = 2\pi f / c`.

    Returns:
        The cumulative 2×2 complex polarimetric transition matrix.
    """
    from differt.em._fresnel import reflection_coefficients
    from differt.em._utils import sp_directions

    vertices = jnp.asarray(vertices)
    normals = jnp.asarray(normals)
    interaction_types = jnp.asarray(interaction_types)
    relative_permittivities = jnp.asarray(relative_permittivities)
    wavenumber = jnp.asarray(wavenumber)

    # Compute ray direction vectors between consecutive vertices
    # [*batch path_length-1 3]
    ray_vectors = jnp.diff(vertices, axis=-2)

    # Normalize to get unit directions and segment lengths
    # [*batch path_length-1 3], [*batch path_length-1 1]
    k_hat, segment_lengths = normalize(ray_vectors, keepdims=True)

    # Incident and reflected/transmitted directions at each interaction
    # k_i: directions arriving at interaction points
    # k_r: directions leaving interaction points
    k_i = k_hat[..., :-1, :]  # [*batch num_interactions 3]
    k_r = k_hat[..., 1:, :]   # [*batch num_interactions 3]

    # Compute s/p directions at each interaction
    # Each returns ((e_i_s, e_i_p), (e_r_s, e_r_p))
    (e_i_s, e_i_p), (e_r_s, e_r_p) = sp_directions(k_i, k_r, normals)

    # Cosine of angle of incidence at each interaction
    cos_theta_i = jnp.abs(jnp.sum(-k_i * normals, axis=-1))  # [*batch num_interactions]

    # Compute Fresnel reflection coefficients
    r_s, r_p = reflection_coefficients(
        jnp.sqrt(relative_permittivities),
        cos_theta_i,
    )

    # Build per-interaction 2×2 transition matrices
    # For reflections: T = [[r_s, 0], [0, r_p]]
    # For now, we only support reflections
    zeros = jnp.zeros_like(r_s)
    # [*batch num_interactions 2 2]
    T_local = jnp.stack([
        jnp.stack([r_s, zeros], axis=-1),
        jnp.stack([zeros, r_p], axis=-1),
    ], axis=-2)

    # Apply phase shift per segment: e^{-j k d}
    # Segment lengths: [*batch path_length-1 1]
    # We need to apply the phase for the segment *after* each interaction
    # i.e., segment from interaction point to next vertex
    segment_d = segment_lengths[..., 1:, 0]  # [*batch num_interactions]
    phase = jnp.exp(-1j * wavenumber * segment_d)

    # Scale transition matrix by phase
    T_local = T_local * phase[..., None, None]

    # Apply spreading factor for spherical wavefronts
    # s_i / (s_i + s_r) where s_i is distance from source to interaction,
    # s_r is distance from interaction to next point
    s_i = segment_lengths[..., :-1, 0]  # [*batch num_interactions]
    s_r = segment_lengths[..., 1:, 0]   # [*batch num_interactions]
    spreading = jnp.sqrt(s_i / jnp.maximum(s_i + s_r, 1e-12))
    T_local = T_local * spreading[..., None, None]

    # Chain all matrices: T_total = T_L @ T_{L-1} @ ... @ T_1
    # We process from last interaction to first
    def scan_fun(
        carry: Complex[Array, "*batch 2 2"],
        T: Complex[Array, "*batch 2 2"],
    ) -> tuple[Complex[Array, "*batch 2 2"], None]:
        return carry @ T, None

    # Initialize with identity
    batch_shape = T_local.shape[:-3]
    identity = jnp.broadcast_to(
        jnp.eye(2, dtype=T_local.dtype),
        (*batch_shape, 2, 2),
    )

    # Transpose interaction axis to scan over it
    # [num_interactions *batch 2 2]
    T_scan = jnp.moveaxis(T_local, -3, 0)

    T_total, _ = jax.lax.scan(scan_fun, identity, T_scan)

    # Apply the initial phase shift from TX to first interaction
    d_tx = segment_lengths[..., 0, 0]  # [*batch]
    phase_tx = jnp.exp(-1j * wavenumber * d_tx)
    T_total = T_total * phase_tx[..., None, None]

    return T_total


@jax.jit
def compute_cir(
    vertices: Float[ArrayLike, "*batch path_length 3"],
    normals: Float[ArrayLike, "*batch num_interactions 3"],
    interaction_types: Int[ArrayLike, "*batch num_interactions"] | None = None,
    relative_permittivities: Complex[ArrayLike, "*batch num_interactions"] | None = None,
    frequency: Float[ArrayLike, ""] = 3e9,
) -> tuple[Complex[Array, "*batch 2 2"], Float[Array, " *batch"]]:
    r"""Compute the polarimetric Channel Impulse Response for a set of traced paths.

    For each path, this function returns:

    1. The 2×2 complex polarimetric channel gain matrix :math:`a_p`
    2. The propagation delay :math:`\tau_p`

    Args:
        vertices: Path vertices from TX to RX.
            Shape ``[*batch, path_length, 3]``.
        normals: Surface normals at each interaction point.
            Shape ``[*batch, path_length - 2, 3]``.
        interaction_types: Type of each interaction.
            If ``None``, all interactions are assumed to be reflections.
        relative_permittivities: Complex relative permittivity per interaction surface.
            If ``None``, all surfaces are assumed to have :math:`\epsilon_r = 5.24`
            (concrete, from ITU-R P.2040).
        frequency: Carrier frequency in Hz.

    Returns:
        A tuple ``(a, tau)`` where:

        - ``a``: Complex 2×2 polarimetric gain matrix per path.
        - ``tau``: Propagation delay in seconds per path.

    Examples:
        Basic CIR computation for a single-reflection path:

        >>> from differt.em._cir import compute_cir
        >>>
        >>> vertices = jnp.array([
        ...     [0.0, 0.0, 2.0],   # TX
        ...     [5.0, 0.0, 0.0],   # Reflection point
        ...     [10.0, 0.0, 2.0],  # RX
        ... ])
        >>> normals = jnp.array([[0.0, 0.0, 1.0]])  # Ground normal
        >>> a, tau = compute_cir(vertices, normals)
        >>> a.shape
        (2, 2)
    """
    vertices = jnp.asarray(vertices)
    normals = jnp.asarray(normals)
    frequency = jnp.asarray(frequency)

    wavenumber = 2 * jnp.pi * frequency / c
    num_interactions = normals.shape[-2]

    if interaction_types is None:
        batch_shape = vertices.shape[:-2]
        interaction_types = jnp.full(
            (*batch_shape, num_interactions),
            InteractionType.REFLECTION,
            dtype=jnp.int32,
        )

    if relative_permittivities is None:
        batch_shape = vertices.shape[:-2]
        relative_permittivities = jnp.full(
            (*batch_shape, num_interactions),
            5.24 + 0j,  # Concrete (ITU-R P.2040)
        )

    # Compute transition matrices
    a = compute_transition_matrices(
        vertices,
        normals,
        interaction_types,
        relative_permittivities,
        wavenumber,
    )

    # Compute propagation delay
    from differt.geometry._utils import path_length

    total_length = path_length(vertices)
    tau = total_length / c

    return a, tau


@jax.jit
def compute_received_power(
    vertices: Float[ArrayLike, "*batch path_length 3"],
    normals: Float[ArrayLike, "*batch num_interactions 3"],
    interaction_types: Int[ArrayLike, "*batch num_interactions"] | None = None,
    relative_permittivities: Complex[ArrayLike, "*batch num_interactions"] | None = None,
    frequency: Float[ArrayLike, ""] = 3e9,
    mask: Float[ArrayLike, " *batch"] | None = None,
) -> Float[Array, ""]:
    r"""Compute total received power from a set of traced paths.

    This is a convenience wrapper around :func:`compute_cir` that:

    1. Computes the CIR for all paths
    2. Sums the power of all valid paths (non-coherent summation)

    Args:
        vertices: Path vertices from TX to RX.
        normals: Surface normals at each interaction point.
        interaction_types: Type of each interaction.
        relative_permittivities: Complex relative permittivity per surface.
        frequency: Carrier frequency in Hz.
        mask: Optional mask indicating which paths are valid.

    Returns:
        Total received power (linear scale).
    """
    a, _tau = compute_cir(
        vertices,
        normals,
        interaction_types,
        relative_permittivities,
        frequency,
    )

    # Power per path: Frobenius norm squared of the gain matrix
    power_per_path = jnp.sum(jnp.abs(a) ** 2, axis=(-2, -1))

    if mask is not None:
        mask = jnp.asarray(mask)
        power_per_path = power_per_path * mask

    return jnp.sum(power_per_path)
