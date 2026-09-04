"""Wavefront curvature state and transport for near-field EM propagation."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from differt.geometry._mesh import Mesh
from differt.geometry._paths import TracedPaths
from differt.geometry._utils import normalize
from differt.utils import safe_divide

from ._interaction_type import InteractionType
from ._utils import _spherical_basis

_ASTIGMATIC_TUPLE_LEN = 4
_CURVATURE_PLANAR_TOL = 1e-12


class WavefrontState(eqx.Module):
    r"""
    Represents the wavefront curvature state along a ray.

    .. note::

        This class is also re-exported directly from the top-level :mod:`differt` package
        (e.g., ``from differt import WavefrontState``).

    The wavefront geometry is described in the plane transverse to the propagation
    direction :math:`\hat{k}` by two principal radii of curvature
    :math:`(\rho_1, \rho_2)` and their corresponding orthogonal unit vectors
    :math:`(\hat{u}_1, \hat{u}_2)`.

    Attributes:
        radii: The principal radii of curvature, shape ``(*batch, 2)``.
        axes: The orthogonal unit vectors defining the principal planes,
            shape ``(*batch, 2, 3)``.
        is_planar: Boolean flags indicating whether each principal curvature
            is zero (infinite radius / plane wave), shape ``(*batch, 2)``.
    """

    radii: Float[Array, "*batch 2"]
    axes: Float[Array, "*batch 2 3"]
    is_planar: Bool[Array, "*batch 2"]

    @classmethod
    def from_tx(
        cls,
        k_hat: Float[Array, "*batch 3"],
        tx_wavefront: Any = 0.0,
    ) -> "WavefrontState":
        r"""
        Create the initial wavefront state at the transmitter.

        Args:
            k_hat: The unit propagation direction leaving the transmitter.
            tx_wavefront: The transmitter wavefront configuration:

                - :data:`None`: Planar wavefront.
                - Scalar or float array: Spherical wavefront with radius :math:`\rho`.
                - ``(rho_s, rho_p)``: Astigmatic wavefront radii, with axes aligned to
                  the local spherical basis :math:`(\hat{\theta}, \hat{\phi})`.
                - ``(rho_s, s_hat, rho_p, p_hat)``: Astigmatic wavefront radii and
                  explicit principal axes.
                - An existing :class:`WavefrontState` instance.

        Returns:
            The initialized wavefront state.
        """
        batch = k_hat.shape[:-1]
        if isinstance(tx_wavefront, cls):
            if tx_wavefront.radii.shape[:-1] == batch:
                return tx_wavefront
            radii = jnp.broadcast_to(tx_wavefront.radii, (*batch, 2))
            axes = jnp.broadcast_to(tx_wavefront.axes, (*batch, 2, 3))
            is_planar = jnp.broadcast_to(tx_wavefront.is_planar, (*batch, 2))
            return cls(radii=radii, axes=axes, is_planar=is_planar)

        theta_hat, phi_hat = _spherical_basis(k_hat)
        default_axes = jnp.stack([theta_hat, phi_hat], axis=-2)

        if tx_wavefront is None:
            radii = jnp.zeros((*batch, 2), dtype=k_hat.dtype)
            axes = default_axes
            is_planar = jnp.ones((*batch, 2), dtype=bool)
            return cls(radii=radii, axes=axes, is_planar=is_planar)

        if isinstance(tx_wavefront, tuple):
            if len(tx_wavefront) == _ASTIGMATIC_TUPLE_LEN:
                rho_s, s_hat, rho_p, p_hat = tx_wavefront
                r_s = jnp.broadcast_to(jnp.asarray(rho_s), batch)
                r_p = jnp.broadcast_to(jnp.asarray(rho_p), batch)
                radii = jnp.stack([r_s, r_p], axis=-1)
                ax_s = jnp.broadcast_to(jnp.asarray(s_hat), (*batch, 3))
                ax_p = jnp.broadcast_to(jnp.asarray(p_hat), (*batch, 3))
                axes = jnp.stack([ax_s, ax_p], axis=-2)
                is_planar = jnp.zeros((*batch, 2), dtype=bool)
                return cls(radii=radii, axes=axes, is_planar=is_planar)

            rho_s, rho_p = tx_wavefront
            r_s = jnp.broadcast_to(jnp.asarray(rho_s), batch)
            r_p = jnp.broadcast_to(jnp.asarray(rho_p), batch)
            radii = jnp.stack([r_s, r_p], axis=-1)
            axes = default_axes
            is_planar = jnp.zeros((*batch, 2), dtype=bool)
            return cls(radii=radii, axes=axes, is_planar=is_planar)

        rho = jnp.broadcast_to(jnp.asarray(tx_wavefront), batch)
        radii = jnp.stack([rho, rho], axis=-1)
        axes = default_axes
        is_planar = jnp.zeros((*batch, 2), dtype=bool)
        return cls(radii=radii, axes=axes, is_planar=is_planar)

    def propagate(
        self,
        distance: Float[Array, "*batch"],
    ) -> "WavefrontState":
        r"""
        Propagate the wavefront along a straight line in free space.

        Args:
            distance: The distance traveled along the ray direction.

        Returns:
            The updated wavefront state.
        """
        dist = distance[..., None]
        new_radii = jnp.where(self.is_planar, self.radii, self.radii + dist)
        batch = jnp.broadcast_shapes(self.radii.shape[:-1], distance.shape)
        new_axes = jnp.broadcast_to(self.axes, (*batch, 2, 3))
        new_is_planar = jnp.broadcast_to(self.is_planar, (*batch, 2))
        return WavefrontState(
            radii=new_radii,
            axes=new_axes,
            is_planar=new_is_planar,
        )

    def reflect(
        self,
        normal: Float[Array, "*batch 3"],
    ) -> "WavefrontState":
        r"""
        Reflect the wavefront off a flat planar interface.

        Args:
            normal: The surface unit normal vector.

        Returns:
            The updated wavefront state.
        """
        norm = normal[..., None, :]
        u_dot_n = jnp.sum(self.axes * norm, axis=-1, keepdims=True)
        new_axes = self.axes - 2.0 * u_dot_n * norm
        new_axes, _ = normalize(new_axes, keepdims=True)
        batch = jnp.broadcast_shapes(self.radii.shape[:-1], normal.shape[:-1])
        new_radii = jnp.broadcast_to(self.radii, (*batch, 2))
        new_is_planar = jnp.broadcast_to(self.is_planar, (*batch, 2))
        return WavefrontState(
            radii=new_radii,
            axes=new_axes,
            is_planar=new_is_planar,
        )

    def transmit(self) -> "WavefrontState":
        r"""
        Transmit the wavefront through a flat, thin slab without deflection.

        Returns:
            The unchanged wavefront state.
        """
        return self

    def diffract(
        self,
        k_in: Float[Array, "*batch 3"],
        k_out: Float[Array, "*batch 3"],
        e_hat: Float[Array, "*batch 3"],
        n0: Float[Array, "*batch 3"] | None = None,
    ) -> tuple["WavefrontState", Float[Array, "*batch 3"]]:
        r"""
        Diffract the wavefront at a straight wedge edge.

        Following Kouyoumjian & Pathak (1974) and McNamara et al. (1990,
        Chapter 6, pp. 264--273, Eq. 6.2--6.5, 6.34, 6.36), straight-edge
        diffraction produces an astigmatic wavefront with one caustic along the edge
        (:math:`\rho_1 = 0`) and second principal radius :math:`\rho_2 = \rho_e^i` equal
        to the incident wavefront's radius in the edge-fixed plane of incidence.

        Args:
            k_in: Unit incident ray direction.
            k_out: Unit diffracted ray direction.
            e_hat: Wedge edge unit tangent vector.
            n0: Optional 0-face surface normal used for canonical edge orientation.

        Returns:
            A tuple of ``(new_state, incident_radii)`` where ``incident_radii`` holds
            ``(rho_1_i, rho_2_i, rho_e_i)``.
        """
        if n0 is not None:
            swap = jnp.sum(k_in * n0, axis=-1, keepdims=True) > 0.0
            e_hat = jnp.where(swap, -e_hat, e_hat)

        phi_prime, _ = normalize(jnp.cross(k_in, e_hat))
        beta_0_prime, _ = normalize(jnp.cross(phi_prime, k_in))

        u1 = self.axes[..., 0, :]
        u2 = self.axes[..., 1, :]
        cos_alpha1 = jnp.sum(beta_0_prime * u1, axis=-1)
        cos_alpha2 = jnp.sum(beta_0_prime * u2, axis=-1)

        rho1 = self.radii[..., 0]
        rho2 = self.radii[..., 1]
        p1 = self.is_planar[..., 0]
        p2 = self.is_planar[..., 1]

        c1 = safe_divide(1.0, rho1)
        c2 = safe_divide(1.0, rho2)
        curv_e = jnp.where(p1, 0.0, cos_alpha1**2 * c1) + jnp.where(
            p2, 0.0, cos_alpha2**2 * c2
        )

        is_planar_e = (p1 & p2) | (curv_e <= _CURVATURE_PLANAR_TOL)
        rho_e = jnp.where(is_planar_e, 0.0, safe_divide(1.0, curv_e))

        phi_d, _ = normalize(jnp.cross(k_out, e_hat))
        phi_d = -phi_d
        beta_0_d, _ = normalize(jnp.cross(phi_d, k_out))

        new_radii = jnp.stack([jnp.zeros_like(rho_e), rho_e], axis=-1)
        new_axes = jnp.stack([phi_d, beta_0_d], axis=-2)
        new_is_planar = jnp.stack([jnp.zeros_like(is_planar_e), is_planar_e], axis=-1)

        new_state = WavefrontState(
            radii=new_radii, axes=new_axes, is_planar=new_is_planar
        )

        rho1_ret = jnp.where(p1, jnp.inf, rho1)
        rho2_ret = jnp.where(p2, jnp.inf, rho2)
        rho_e_ret = jnp.where(is_planar_e, jnp.inf, rho_e)
        incident_radii = jnp.stack([rho1_ret, rho2_ret, rho_e_ret], axis=-1)

        return new_state, incident_radii


class PathWavefront(eqx.Module):
    r"""
    Wavefront propagation history along traced paths.

    Attributes:
        state: The final wavefront state at the receiver.
        incident_radii: The incident radii ``(rho_1_i, rho_2_i, rho_e_i)``
            at each interaction bounce, shape ``(*batch, order, 3)``.
        spreading_factor: The accumulated field amplitude spreading factor along each path,
            shape ``(*batch,)``.
        segment_radii: The principal radii at the start of each path segment,
            shape ``(*batch, num_segments, 2)``.
    """

    state: WavefrontState
    incident_radii: Float[Array, "*batch order 3"]
    spreading_factor: Float[Array, " *batch"]
    segment_radii: Float[Array, "*batch num_segments 2"]


@eqx.filter_jit
def propagate_wavefront(
    paths: TracedPaths,
    mesh: Mesh,
    tx_wavefront: Any = 0.0,
) -> PathWavefront:
    r"""
    Propagate the wavefront curvature state along the given traced paths.

    .. note::

        This function is also re-exported directly from the top-level :mod:`differt` package
        (e.g., ``from differt import propagate_wavefront``).

    Performs a scan along each path's segments and interaction bounces, transporting
    principal radii and axes through free space, reflections, transmissions, and
    diffractions.

    Args:
        paths: The traced paths.
        mesh: The scene triangle mesh.
        tx_wavefront: The transmitter wavefront curvature configuration.

    Returns:
        A :class:`PathWavefront` instance carrying the final state, per-bounce incident
        curvatures, segment radii, and total spreading factor.
    """
    path_segments = jnp.diff(paths.vertices, axis=-2)
    k_hat, s = normalize(path_segments, keepdims=True)
    s = s[..., 0]
    batch = paths.shape
    order = paths.order

    state_0 = WavefrontState.from_tx(k_hat[..., 0, :], tx_wavefront)

    if order == 0:
        s_0 = s[..., 0]
        final_state = state_0.propagate(s_0)
        is_planar_all = jnp.all(state_0.is_planar, axis=-1)
        rho_s = state_0.radii[..., 0]
        rho_p = state_0.radii[..., 1]
        spreading = jnp.where(
            is_planar_all,
            jnp.ones_like(s_0),
            safe_divide(1.0, jnp.sqrt((s_0 + rho_s) * (s_0 + rho_p))),
        )
        incident_radii = jnp.zeros((*batch, 0, 3), dtype=paths.vertices.dtype)
        segment_radii = state_0.radii[..., None, :]
        return PathWavefront(
            state=final_state,
            incident_radii=incident_radii,
            spreading_factor=spreading,
            segment_radii=segment_radii,
        )

    n0_he, _, e_hat_he, _ = mesh._wedge_static_geometry()  # ruff: ignore[private-member-access]
    obj_indices = jnp.clip(paths.objects[..., 1:-1], 0, mesh.num_triangles - 1)
    face_normals = jnp.take(mesh.normals, obj_indices, axis=0)

    half_edge_idx = paths.objects[..., 1:-1]
    prim0_idx = half_edge_idx // 3
    local_edge_idx = half_edge_idx % 3
    e_hat_arr = e_hat_he[prim0_idx, local_edge_idx]
    n0_arr = n0_he[prim0_idx, local_edge_idx]

    seg_radii_list = [state_0.radii]
    inc_radii_list = []

    curr_state = state_0.propagate(s[..., 0])

    for j in range(order):
        kind = paths.interaction_types[..., j]
        k_in_j = k_hat[..., j, :]
        k_out_j = k_hat[..., j + 1, :]
        norm_j = face_normals[..., j, :]
        edge_j = e_hat_arr[..., j, :]
        n0_j = n0_arr[..., j, :]

        refl_state = curr_state.reflect(norm_j)
        diff_state, inc_radii_j = curr_state.diffract(k_in_j, k_out_j, edge_j, n0_j)
        inc_radii_list.append(inc_radii_j)

        # Map InteractionType values [-1, 0, 1, 2, 3, 4] to branch index [0..5]:
        # 0: NONE (pass-through / inactive candidate)
        # 1: REFLECTION (mirror reflection of axes, radii/planar unchanged)
        # 2: DIFFRACTION (astigmatic edge diffraction)
        # 3: SCATTERING (radii reset to 0, planar=False, axes unchanged)
        # 4: TRANSMISSION (pass-through / unchanged)
        # 5: RIS (specular reflection of axes, radii/planar unchanged)
        which = jnp.clip(kind + 1, 0, 5)
        which_radii = jnp.broadcast_to(which[..., None], curr_state.radii.shape)
        which_axes = jnp.broadcast_to(which[..., None, None], curr_state.axes.shape)

        new_radii = jax.lax.select_n(
            which_radii,
            curr_state.radii,
            curr_state.radii,
            diff_state.radii,
            jnp.zeros_like(curr_state.radii),
            curr_state.radii,
            curr_state.radii,
        )
        new_is_planar = jax.lax.select_n(
            which_radii,
            curr_state.is_planar,
            curr_state.is_planar,
            diff_state.is_planar,
            jnp.zeros_like(curr_state.is_planar),
            curr_state.is_planar,
            curr_state.is_planar,
        )
        new_axes = jax.lax.select_n(
            which_axes,
            curr_state.axes,
            refl_state.axes,
            diff_state.axes,
            curr_state.axes,
            curr_state.axes,
            refl_state.axes,
        )

        out_state = WavefrontState(
            radii=new_radii, axes=new_axes, is_planar=new_is_planar
        )
        seg_radii_list.append(out_state.radii)
        curr_state = out_state.propagate(s[..., j + 1])

    final_state = curr_state
    segment_radii = jnp.stack(seg_radii_list, axis=-2)
    incident_radii = jnp.stack(inc_radii_list, axis=-2)

    s_tot = jnp.sum(s, axis=-1)
    is_planar_tx = jnp.all(state_0.is_planar, axis=-1)
    rho_s = state_0.radii[..., 0]
    rho_p = state_0.radii[..., 1]

    spreading_go = jnp.where(
        is_planar_tx,
        jnp.ones_like(s_tot),
        safe_divide(1.0, jnp.sqrt((s_tot + rho_s) * (s_tot + rho_p))),
    )

    is_diffraction = paths.interaction_types == InteractionType.DIFFRACTION
    has_diffraction = jnp.any(is_diffraction, axis=-1)

    diff_idx = jnp.argmax(is_diffraction, axis=-1)
    seg_idx = jnp.arange(s.shape[-1])
    is_after = seg_idx > diff_idx[..., None]
    s_after = jnp.sum(jnp.where(is_after, s, 0.0), axis=-1)

    rho_1_i = jnp.take_along_axis(incident_radii[..., 0], diff_idx[..., None], axis=-1)[
        ..., 0
    ]
    rho_2_i = jnp.take_along_axis(incident_radii[..., 1], diff_idx[..., None], axis=-1)[
        ..., 0
    ]
    rho_e_i = jnp.take_along_axis(incident_radii[..., 2], diff_idx[..., None], axis=-1)[
        ..., 0
    ]

    rho_1_safe = jnp.where(is_planar_tx, 1.0, rho_1_i)
    rho_2_safe = jnp.where(is_planar_tx, 1.0, rho_2_i)
    rho_e_safe = jnp.where(is_planar_tx, 1.0, rho_e_i)

    denom = (
        jnp.maximum(rho_1_safe, 1e-12)
        * jnp.maximum(rho_2_safe, 1e-12)
        * jnp.maximum(s_after, 1e-12)
        * jnp.maximum(rho_e_safe + s_after, 1e-12)
    )
    spreading_diff_astigmatic = safe_divide(
        jnp.sqrt(jnp.maximum(rho_e_safe, 1e-12)),
        jnp.sqrt(denom),
    )
    spreading_diff_planar = safe_divide(1.0, jnp.sqrt(s_after))
    spreading_diff = jnp.where(
        is_planar_tx, spreading_diff_planar, spreading_diff_astigmatic
    )

    spreading = jnp.where(has_diffraction, spreading_diff, spreading_go)

    return PathWavefront(
        state=final_state,
        incident_radii=incident_radii,
        spreading_factor=spreading,
        segment_radii=segment_radii,
    )
