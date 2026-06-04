import functools
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Complex, Float

from differt.geometry import Paths, TriangleMesh, normalize
from differt.utils import safe_divide

from ._constants import c, epsilon_0, z_0
from ._fresnel import reflection_coefficients
from ._material import Material, materials
from ._utils import sp_directions, sp_rotation_matrix


@jax.jit
def _get_reflection_coefficients(
    n_r: Complex[Array, "*batch"],
    cos_theta_i: Float[Array, "*batch"],
    thickness: Float[Array, "*batch"],
    wavelength: Float[ArrayLike, ""],
) -> tuple[Complex[Array, "*batch"], Complex[Array, "*batch"]]:
    r_s_inf, r_p_inf = reflection_coefficients(n_r, cos_theta_i)

    eta = n_r**2
    sin_theta_sqr = 1.0 - cos_theta_i**2
    a = jnp.sqrt(eta - sin_theta_sqr)

    q = (2.0 * jnp.pi * thickness / wavelength) * a
    exp_j_2q = jnp.exp(-2j * q)

    r_s_slab = safe_divide(r_s_inf * (1.0 - exp_j_2q), 1.0 - r_s_inf**2 * exp_j_2q)
    r_p_slab = safe_divide(r_p_inf * (1.0 - exp_j_2q), 1.0 - r_p_inf**2 * exp_j_2q)

    use_slab = thickness >= 0.0
    r_s = jnp.where(use_slab, r_s_slab, r_s_inf)
    r_p = jnp.where(use_slab, r_p_slab, r_p_inf)
    return r_s, r_p


@jax.jit
def _spherical_basis(
    k: Float[Array, "*batch 3"],
) -> tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]]:
    x = k[..., 0]
    y = k[..., 1]
    z = jnp.clip(k[..., 2], -1.0, 1.0)
    theta = jnp.arccos(z)
    phi = jnp.arctan2(y, x)

    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)

    theta_hat = jnp.stack(
        [cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta], axis=-1
    )
    phi_hat = jnp.stack([-sin_phi, cos_phi, jnp.zeros_like(phi)], axis=-1)
    return theta_hat, phi_hat


def compute_received_fields(
    paths: Paths,
    mesh: TriangleMesh,
    frequency: Float[ArrayLike, ""],
    tx_polarization: Any = "V",
    rx_polarization: Any = "V",
    radio_materials: Mapping[str, Material] | None = None,
) -> Complex[Array, "*batch"]:
    """
    Compute the received complex fields for each path.

    Args:
        paths: The paths.
        mesh: The triangle mesh of the scene.
        frequency: The operating frequency in Hz.
        tx_polarization: The transmitter antenna polarization or pattern.
        rx_polarization: The receiver antenna polarization or pattern.
        radio_materials: The dictionary of material properties.

    Returns:
        The received complex fields of shape ``*batch``.

    Raises:
        ValueError: If the mesh does not contain face materials when order > 0.
    """
    if radio_materials is None:
        radio_materials = materials

    # Extract material parameters
    eta_r = jnp.array([
        radio_materials[mat_name].relative_permittivity(frequency)
        for mat_name in mesh.material_names
    ])
    conductivity = jnp.array([
        radio_materials[mat_name].conductivity(frequency)
        for mat_name in mesh.material_names
    ])
    thickness = jnp.array([
        radio_materials[mat_name].thickness
        if radio_materials[mat_name].thickness is not None
        else -1.0
        for mat_name in mesh.material_names
    ])
    omega = 2.0 * jnp.pi * frequency
    epsilon_complex = eta_r - 1j * conductivity / (omega * epsilon_0)
    n_complex = jnp.sqrt(epsilon_complex)
    wavelength = c / frequency

    # Get path segments
    path_segments = jnp.diff(paths.vertices, axis=-2)
    k, s = normalize(path_segments, keepdims=True)

    # Spherical basis unit vectors for all segments
    theta_hat_arr, phi_hat_arr = _spherical_basis(k)

    # Initial field e_field
    theta_hat_0 = theta_hat_arr[..., 0, :]
    phi_hat_0 = phi_hat_arr[..., 0, :]

    # Check if tx_polarization is a BaseAntenna subclass
    if hasattr(tx_polarization, "fields"):
        T = paths.vertices[..., 0, :]
        r_hat = k[..., 0, :]
        e_init, _ = tx_polarization.fields(T + r_hat)
        e_dir = e_init * jnp.exp(1j * tx_polarization.wavenumber)
        e_theta = jnp.sum(e_dir * theta_hat_0, axis=-1)
        e_phi = jnp.sum(e_dir * phi_hat_0, axis=-1)
        e_field = jnp.stack([e_theta, e_phi], axis=-1)
    elif tx_polarization == "V":
        e_field = jnp.stack(
            [jnp.ones(theta_hat_0.shape[:-1]), jnp.zeros(theta_hat_0.shape[:-1])],
            axis=-1,
        ).astype(complex)
    elif tx_polarization == "H":
        e_field = jnp.stack(
            [jnp.zeros(theta_hat_0.shape[:-1]), jnp.ones(theta_hat_0.shape[:-1])],
            axis=-1,
        ).astype(complex)
    else:
        p = jnp.asarray(tx_polarization, dtype=complex)
        p_dot_theta = jnp.sum(p * theta_hat_0, axis=-1)
        p_dot_phi = jnp.sum(p * phi_hat_0, axis=-1)
        e_field = jnp.stack([p_dot_theta, p_dot_phi], axis=-1)

    e_field_vec = e_field[..., None]  # shape [..., 2, 1]

    # Process reflections if order > 0
    if paths.order > 0:
        if mesh.face_materials is None:
            msg = "Mesh must contain face materials to compute reflections."
            raise ValueError(msg)

        obj_indices = paths.objects[..., 1:-1]
        mat_indices = jnp.take(mesh.face_materials, obj_indices, axis=0)
        obj_normals = jnp.take(mesh.normals, obj_indices, axis=0)

        k_in = k[..., :-1, :]
        k_out = k[..., 1:, :]
        n = obj_normals

        n_r_val = jnp.take(n_complex, mat_indices, axis=0)
        thickness_val = jnp.take(thickness, mat_indices, axis=0)

        (e_i_s, e_i_p), (e_r_s, e_r_p) = sp_directions(k_in, k_out, n)
        cos_theta_i = jnp.sum(n * -k_in, axis=-1)

        r_s, r_p = _get_reflection_coefficients(
            n_r_val, cos_theta_i, thickness_val, wavelength
        )

        theta_in = theta_hat_arr[..., :-1, :]
        phi_in = phi_hat_arr[..., :-1, :]
        theta_out = theta_hat_arr[..., 1:, :]
        phi_out = phi_hat_arr[..., 1:, :]

        in_rot = sp_rotation_matrix(theta_in, phi_in, e_i_s, e_i_p)
        out_rot = sp_rotation_matrix(e_r_s, e_r_p, theta_out, phi_out)

        zero = jnp.zeros_like(r_s)
        d_j = jnp.stack(
            [jnp.stack([r_s, zero], axis=-1), jnp.stack([zero, r_p], axis=-1)],
            axis=-2,
        )

        j_mat = jnp.matmul(out_rot, jnp.matmul(d_j, in_rot))

        # Multiply transition matrices sequentially along the order dimension
        j_list = [j_mat[..., j, :, :] for j in range(paths.order)]
        j_total = functools.reduce(lambda x, y: jnp.matmul(y, x), j_list)
        e_field_vec = jnp.matmul(j_total, e_field_vec)
        e_field = e_field_vec[..., 0]

    # Project final field onto receiver polarization
    theta_hat_last = theta_hat_arr[..., -1, :]
    phi_hat_last = phi_hat_arr[..., -1, :]

    if hasattr(rx_polarization, "fields"):
        r = paths.vertices[..., -1, :]
        k_last = k[..., -1, :]
        e_rx, _ = rx_polarization.fields(r - k_last)
        e_rx_dir = e_rx * jnp.exp(1j * rx_polarization.wavenumber)
        u_theta = jnp.sum(e_rx_dir * theta_hat_last, axis=-1)
        u_phi = jnp.sum(e_rx_dir * phi_hat_last, axis=-1)
        u = jnp.stack([u_theta, u_phi], axis=-1)
    elif rx_polarization == "V":
        theta_hat_neg_k_last = _spherical_basis(-k[..., -1, :])[0]
        a_coeff = jnp.sum(theta_hat_last * theta_hat_neg_k_last, axis=-1)
        u = jnp.stack([a_coeff, jnp.zeros_like(a_coeff)], axis=-1)
    elif rx_polarization == "H":
        theta_hat_neg_k_last = _spherical_basis(-k[..., -1, :])[0]
        a_coeff = jnp.sum(theta_hat_last * theta_hat_neg_k_last, axis=-1)
        u = jnp.stack([jnp.zeros_like(a_coeff), -a_coeff], axis=-1)
    else:
        p = jnp.asarray(rx_polarization)
        p_dot_theta = jnp.sum(p * theta_hat_last, axis=-1)
        p_dot_phi = jnp.sum(p * phi_hat_last, axis=-1)
        u = jnp.stack([p_dot_theta, p_dot_phi], axis=-1)

    a_r = jnp.sum(u * e_field, axis=-1)

    # Spreading factor and phase shift
    s_tot = s.sum(axis=-2)
    spreading_factor = safe_divide(1.0, s_tot)
    phase_val = -2.0 * jnp.pi * frequency * s_tot / c
    phase_shift = jax.lax.complex(jnp.cos(phase_val), jnp.sin(phase_val))

    a_r *= (spreading_factor * phase_shift)[..., 0]
    a = a_r * (wavelength / (4 * jnp.pi))

    # Apply path mask if present
    if paths.mask is not None:
        a = a * paths.mask

    return a


@functools.partial(jax.jit, static_argnames=("coherent", "axis"))
def compute_received_power(
    fields: Complex[Array, "*batch"],
    z_0: Float[ArrayLike, ""] | float = z_0,
    coherent: bool = True,
    axis: int | None = None,
) -> Float[Array, "..."]:
    """
    Compute the received power from the received fields (in dBW).

    Args:
        fields: The complex received fields.
        z_0: The reference impedance.
        coherent: Whether to sum coherently (vector sum of fields before power)
            or non-coherently (power sum of individual fields).
            Only active if ``axis`` is not None.
        axis: The axis along which to sum the fields. If None, no sum is performed.

    Returns:
        The received power in dBW.
    """
    if axis is not None:
        if coherent:
            summed_fields = jnp.sum(fields, axis=axis)
            power = jnp.abs(summed_fields) ** 2 / z_0
        else:
            power = jnp.sum(jnp.abs(fields) ** 2 / z_0, axis=axis)
    else:
        power = jnp.abs(fields) ** 2 / z_0
    return 10.0 * jnp.log10(power)


def compute_cir(
    paths: Paths,
    fields: Complex[Array, "*batch"],
) -> tuple[Float[Array, "*batch"], Complex[Array, "*batch"]]:
    """
    Compute the Channel Impulse Response (CIR) as (delay, fields) pairs.

    Args:
        paths: The paths.
        fields: The complex received fields.

    Returns:
        A tuple of (delay, fields) where delay and fields have the same shape.
    """
    path_segments = jnp.diff(paths.vertices, axis=-2)
    lengths = jnp.linalg.norm(path_segments, axis=-1).sum(axis=-1)
    delay = lengths / c
    return delay, fields
