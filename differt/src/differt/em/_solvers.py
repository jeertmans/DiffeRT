import abc
import functools
from collections.abc import Mapping
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Complex, Float, Inexact, Int

from differt.geometry._mesh import Mesh
from differt.geometry._paths import TracedPaths
from differt.geometry._utils import normalize
from differt.utils import safe_divide

from ._constants import c, epsilon_0
from ._fresnel import reflection_coefficients, slab_coefficients
from ._interaction_type import InteractionType
from ._material import Material, materials
from ._utd import diffraction_coefficients
from ._utils import sp_directions, sp_rotation_matrix


def _wavefront_radii(
    tx_wavefront_radii: Float[ArrayLike, "*#batch"]
    | tuple[Float[ArrayLike, "*#batch"], Float[ArrayLike, "*#batch"]]
    | tuple[
        Float[ArrayLike, "*#batch"],
        Float[ArrayLike, "*#batch 3"],
        Float[ArrayLike, "*#batch"],
        Float[ArrayLike, "*#batch 3"],
    ]
    | None,
) -> tuple[Float[Array, "*batch"], Float[Array, "*batch"]] | None:
    """
    Split a wavefront-radii argument into its two principal (s- and p-plane) radii.

    A single value describes an isotropic (spherical) wavefront, i.e., both
    principal radii equal that value; a ``(rho_s, rho_p)`` tuple describes a
    general astigmatic wavefront, with each element broadcastable against
    the paths' batch dimensions; :data:`None` describes a planar wavefront
    (this is a static, Python-level choice, not a traced value, so it is
    safe to branch on under :func:`jax.jit`). A ``(rho_s, s_hat, rho_p,
    p_hat)`` 4-tuple, as returned by
    :meth:`AbstractAntenna.wavefront_radii<differt.em.AbstractAntenna.wavefront_radii>`,
    is also accepted, for convenience; ``s_hat``/``p_hat`` are dropped, as
    no formula using this helper's output currently consumes wavefront
    orientation.

    Returns:
        The ``(rho_s, rho_p)`` pair of principal radii, or :data:`None`
        for a planar wavefront.
    """
    if tx_wavefront_radii is None:
        return None
    if isinstance(tx_wavefront_radii, tuple):
        if len(tx_wavefront_radii) == 4:  # ruff:ignore[magic-value-comparison]
            rho_s, _s_hat, rho_p, _p_hat = tx_wavefront_radii
        else:
            rho_s, rho_p = tx_wavefront_radii
        return jnp.asarray(rho_s), jnp.asarray(rho_p)
    rho = jnp.asarray(tx_wavefront_radii)
    return rho, rho


@jax.jit
def _get_reflection_coefficients(
    n_r: Complex[Array, "*batch"],
    cos_theta_i: Float[Array, "*batch"],
    thickness: Float[Array, "*batch"],
    wavelength: Float[ArrayLike, "*#batch"],
) -> tuple[Complex[Array, "*batch"], Complex[Array, "*batch"]]:
    """
    Reflection off a slab, or an infinite half-space if ``thickness < 0`` (sentinel).

    Returns:
        The s and p reflection coefficients.
    """
    r_s_inf, r_p_inf = reflection_coefficients(n_r, cos_theta_i)
    (r_s_slab, r_p_slab), _ = slab_coefficients(
        n_r, cos_theta_i, jnp.maximum(thickness, 0.0), wavelength
    )

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


def _material_arrays(
    mesh: Mesh,
    radio_materials: Mapping[str, Material],
    frequency: Float[ArrayLike, "*#batch"],
) -> tuple[Complex[Array, "*batch num_materials"], Float[Array, " num_materials"]]:
    """Return per-material complex refractive index and thickness arrays.

    The refractive index array has an extra (leading) batch dimension if
    ``frequency`` is not a scalar, since permittivity and conductivity are
    frequency-dependent; thickness is not.
    """
    frequency = jnp.asarray(frequency)
    eta_r = jnp.stack(
        [
            radio_materials[mat_name].relative_permittivity(frequency)
            for mat_name in mesh.material_names
        ],
        axis=-1,
    )
    conductivity = jnp.stack(
        [
            radio_materials[mat_name].conductivity(frequency)
            for mat_name in mesh.material_names
        ],
        axis=-1,
    )
    thickness = jnp.array([
        radio_materials[mat_name].thickness
        if radio_materials[mat_name].thickness is not None
        else -1.0
        for mat_name in mesh.material_names
    ])
    omega = 2.0 * jnp.pi * frequency
    epsilon_complex = eta_r - 1j * conductivity / (omega[..., None] * epsilon_0)
    n_complex = jnp.sqrt(epsilon_complex)
    return n_complex, thickness


def _scattering_properties(
    mesh: Mesh,
    radio_materials: Mapping[str, Material],
) -> tuple[Float[Array, " num_materials"], Float[Array, " num_materials"]]:
    """Return per-material (scattering_coefficient, xpd_coefficient) arrays."""
    scattering_coefficient = jnp.array([
        radio_materials[mat_name].scattering_coefficient
        for mat_name in mesh.material_names
    ])
    xpd_coefficient = jnp.array([
        radio_materials[mat_name].xpd_coefficient for mat_name in mesh.material_names
    ])
    return scattering_coefficient, xpd_coefficient


def _scattering_pattern_values(
    mesh: Mesh,
    radio_materials: Mapping[str, Material],
    k_in: Float[Array, "*batch order 3"],
    k_out: Float[Array, "*batch order 3"],
    obj_normals: Float[Array, "*batch order 3"],
    mat_indices: Int[Array, "*batch order"],
) -> Float[Array, "*batch order"]:
    """Evaluate each material's (possibly distinct) scattering pattern on the shared per-bounce geometry, then select the per-bounce value.

    Each material may define its own
    :attr:`Material.scattering_pattern<differt.em._material.Material.scattering_pattern>`
    callable, so (unlike a plain per-material scalar) this cannot be
    reduced to a single array lookup: every material's pattern is
    evaluated on the full per-bounce geometry, then the result for the
    material actually hit at each bounce is selected.

    Returns:
        The per-bounce scattering pattern value.
    """
    f_s_per_material = jnp.stack(
        [
            jnp.asarray(
                radio_materials[mat_name].scattering_pattern(k_in, k_out, obj_normals)
            )
            for mat_name in mesh.material_names
        ],
        axis=-1,
    )
    return jnp.take_along_axis(f_s_per_material, mat_indices[..., None], axis=-1)[
        ..., 0
    ]


def _take_material_property(
    prop: Inexact[ArrayLike, "*#batch num_materials"],
    mat_indices: Int[Array, "*batch order"],
) -> Array:
    """Gather a per-material property array at the given per-bounce material indices.

    Unlike a plain :func:`jax.numpy.take`, this correctly broadcasts a
    ``prop`` array that carries its own (e.g., frequency-dependent) batch
    dimension against the paths' batch dimension, before selecting one
    value per bounce.

    Returns:
        The gathered per-bounce property array.
    """
    prop = jnp.broadcast_to(
        jnp.asarray(prop), (*mat_indices.shape[:-1], jnp.shape(prop)[-1])
    )
    gathered = jnp.take_along_axis(
        jnp.broadcast_to(prop[..., None, :], (*mat_indices.shape, prop.shape[-1])),
        mat_indices[..., None],
        axis=-1,
    )
    return gathered[..., 0]


def _surface_interaction_geometry(
    paths: TracedPaths,
    mesh: Mesh,
    frequency: Float[Array, "*#batch"],
    radio_materials: Mapping[str, Material],
) -> tuple[
    Float[Array, "*batch num_segments 3"],
    Float[Array, "*batch order 3"],
    Float[Array, "*batch order 3"],
    Complex[Array, "*batch order"],
    Float[Array, "*batch order"],
    Float[Array, "*batch order"],
    Float[Array, "*#batch"],
]:
    """
    Shared per-bounce geometry/material extraction for surface interactions.

    Returns:
        A tuple of ``(k, k_in, obj_normals, n_r_val, thickness_val, cos_theta_i, wavelength)``.

    Raises:
        ValueError: If the mesh does not contain face materials.
    """
    if mesh.face_materials is None:
        msg = "Mesh must contain face materials to compute surface interactions."
        raise ValueError(msg)

    n_complex, thickness = _material_arrays(mesh, radio_materials, frequency)

    # This method may run on bounces that are not actually a REFLECTION or
    # TRANSMISSION interaction (its result is discarded later, based on
    # 'interaction_types') -- in particular, for a DIFFRACTION bounce,
    # 'objects' holds a half-edge index, which is generally out of bounds
    # for a triangle-indexed array; clip it to avoid NaN-filled out-of-bounds
    # gathers, which would otherwise poison the 'jnp.where'-based
    # combination in 'GeometricFieldSolver.transition_matrices'.
    obj_indices = jnp.clip(paths.objects[..., 1:-1], 0, mesh.num_triangles - 1)
    mat_indices = jnp.take(mesh.face_materials, obj_indices, axis=0)
    obj_normals = jnp.take(mesh.normals, obj_indices, axis=0)

    path_segments = jnp.diff(paths.vertices, axis=-2)
    k, _ = normalize(path_segments, keepdims=True)
    k_in = k[..., :-1, :]

    n_r_val = _take_material_property(n_complex, mat_indices)
    thickness_val = jnp.take(thickness, mat_indices, axis=0)
    cos_theta_i = jnp.sum(obj_normals * -k_in, axis=-1)
    wavelength = c / frequency

    return k, k_in, obj_normals, n_r_val, thickness_val, cos_theta_i, wavelength


def _wedge_static_geometry(
    mesh: Mesh,
) -> tuple[
    Float[Array, "num_triangles 3 3"],
    Float[Array, "num_triangles 3 3"],
    Float[Array, "num_triangles 3 3"],
    Int[Array, "num_triangles 3"],
]:
    r"""
    Compute the path-independent (canonical) wedge geometry for every half-edge.

    Every triangle edge (whether or not it is an actual diffraction edge)
    is addressed as a ``(triangle, local_edge)`` half-edge pair, matching
    Sionna RT's own wedge addressing
    (``sionna.rt.utils.wedges.wedge_geometry``) and, unlike
    :attr:`Mesh.diffraction_edges<differt.geometry.Mesh.diffraction_edges>`
    (which deduplicates edges via a non-:func:`jax.jit`-compatible
    :func:`jax.numpy.unique`), stays :func:`jax.jit`-compatible and never
    has a zero-sized axis.

    Follows the same orientation convention as
    ``wedge_geometry``: the face normals are oriented so that the
    interior angle of the wedge is at most :math:`\pi`, and the edge
    direction is oriented such that ``cross(n0, e_hat)`` points toward
    the (arbitrarily labeled) 0-face. Half-edges that are not actual
    diffraction edges (mesh boundary, or coplanar/inactive neighbors, see
    :attr:`Mesh.diffraction_edges_mask<differt.geometry.Mesh.diffraction_edges_mask>`)
    get an arbitrary, self-consistent (but physically meaningless)
    placeholder geometry; callers are expected to discard those entries.

    Returns:
        A tuple of ``(n0, nn, e_hat, primn)`` (``prim0`` is implicitly
        ``arange(num_triangles)[:, None]`` broadcast against the local-edge
        axis).
    """
    adj_t, _ = mesh._connectivity()  # ruff:ignore[private-member-access]
    num_triangles = mesh.num_triangles
    primn = jnp.where(adj_t == -1, jnp.arange(num_triangles)[:, None], adj_t)
    prim0 = jnp.broadcast_to(jnp.arange(num_triangles)[:, None], primn.shape)

    e0, e1 = mesh.triangle_edges[..., 0, :], mesh.triangle_edges[..., 1, :]

    normals = mesh.normals
    n0_raw = jnp.take(normals, prim0, axis=0)
    nn_raw = jnp.take(normals, primn, axis=0)

    triangle_vertices = mesh.triangle_vertices
    f0 = jnp.mean(jnp.take(triangle_vertices, prim0, axis=0), axis=-2)
    fn = jnp.mean(jnp.take(triangle_vertices, primn, axis=0), axis=-2)

    flip_n0 = jnp.sum(n0_raw * (fn - e0), axis=-1) > 0.0
    n0 = jnp.where(flip_n0[..., None], -n0_raw, n0_raw)
    flip_nn = jnp.sum(nn_raw * (f0 - e0), axis=-1) > 0.0
    nn = jnp.where(flip_nn[..., None], -nn_raw, nn_raw)

    e_hat_raw, _ = normalize(e1 - e0)
    t0 = jnp.cross(n0, e_hat_raw)
    flip_e = jnp.sum(t0 * (f0 - e0), axis=-1) < 0.0
    e_hat = jnp.where(flip_e[..., None], -e_hat_raw, e_hat_raw)

    return n0, nn, e_hat, primn


class AbstractFieldSolver(eqx.Module):
    """
    Abstract base class for all EM field solvers.

    A field solver computes the received complex field(s) from a set of
    paths (however they were obtained) and the geometry/materials they
    interacted with.

    This mirrors the path solver hierarchy
    (:class:`AbstractPathSolver<differt.geometry.AbstractPathSolver>`):
    subclasses are expected to target a specific kind of paths, e.g.,
    :class:`GeometricFieldSolver` for
    :class:`TracedPaths<differt.geometry.TracedPaths>`, or, in the future,
    a solver for :class:`LaunchedPaths<differt.geometry.LaunchedPaths>`.
    """

    @abc.abstractmethod
    def compute_fields(
        self,
        paths: Any,
        mesh: Mesh,
        frequency: Float[ArrayLike, "*#batch"],
    ) -> Complex[Array, "*batch"]:
        """
        Compute the received complex fields for the given paths.

        Solver-specific configuration (e.g., antenna polarization, radio
        materials) belongs on the solver instance itself, as attributes,
        rather than as extra arguments here -- see, e.g.,
        :class:`GeometricFieldSolver`'s attributes.

        Args:
            paths: The paths.
            mesh: The triangle mesh of the scene.
            frequency: The operating frequency (or frequencies) in Hz.

        Returns:
            The received complex fields.
        """


class GeometricFieldSolver(AbstractFieldSolver):
    r"""
    Computes fields for :class:`TracedPaths<differt.geometry.TracedPaths>`.

    This solver combines a transmitter excitation, one 2x2 dyadic
    transition (Jones) matrix per interaction along each path, and a
    receiver projection, following Geometrical Optics (GO) and the
    Uniform Theory of Diffraction (UTD), see :mod:`differt.em`.

    To support an additional :class:`InteractionType`, subclass this
    solver, override the matching ``*_matrix`` method (e.g.,
    :meth:`diffraction_matrix`), and add the corresponding member to
    :attr:`supported_interaction_types`. Both steps are required: the
    latter is what lets :meth:`transition_matrices` stay
    :func:`jax.jit`-compatible, as it decides, from a plain Python
    (i.e., non-traced) set, which of the ``*_matrix`` methods to call,
    rather than branching on the (traced) contents of
    :attr:`TracedPaths.interaction_types<differt.geometry.TracedPaths.interaction_types>`.

    Solver-specific configuration (:attr:`tx_polarization`,
    :attr:`rx_polarization`, :attr:`radio_materials`,
    :attr:`tx_wavefront_radii`) lives on the solver instance itself, as
    plain attributes -- mirroring
    :class:`AbstractPathSolver<differt.geometry.AbstractPathSolver>` and
    its subclasses (e.g.,
    :class:`ExhaustivePathTracer<differt.geometry.ExhaustivePathTracer>`)
    -- rather than being passed as keyword arguments at call time; every
    ``*_matrix`` method, :meth:`transition_matrices`,
    :meth:`compute_fields`, and :meth:`spreading_factor` therefore only
    take ``(paths, mesh, frequency)`` (the per-call data), reading their
    configuration from ``self``.

    Examples:
        .. code-block:: python

            from differt.em import (
                GeometricFieldSolver,
                InteractionType,
                compute_received_fields,
            )


            class MyFieldSolver(GeometricFieldSolver):
                supported_interaction_types = frozenset({
                    InteractionType.REFLECTION,
                    InteractionType.DIFFRACTION,
                })

                def diffraction_matrix(
                    self, paths, mesh, frequency
                ): ...  # your implementation, using 'self.radio_materials' etc.


            fields = compute_received_fields(
                paths, mesh, frequency, solver=MyFieldSolver()
            )

    Note:
        Unlike Sionna RT, which only supports a point source (equivalently,
        a wavefront that is already spherical, with a radius of curvature
        equal to the geometric distance traveled), this solver supports a
        **non-planar** incident wavefront at the transmitter, via the
        :attr:`tx_wavefront_radii` attribute, used internally by
        :meth:`compute_fields`,
        :meth:`transition_matrices`, :meth:`diffraction_matrix`, and
        :meth:`spreading_factor` (all overridable). It accepts either a
        single value, for an isotropic (spherical) wavefront, a
        ``(rho_s, rho_p)`` tuple, for a general **astigmatic** wavefront
        with unequal principal radii along the s- and p-planes, or
        :data:`None`, for a **planar** wavefront: a chain of flat-mirror
        reflections and/or transmissions leaves both principal radii
        unchanged (only the *total* path length, plus each of
        ``rho_s``/``rho_p``, matters, regardless of order), since neither
        interaction does anything but fold the ray direction (for a flat
        mirror) or leave it unchanged (for a thin transmitting slab).

        An astigmatic ``tx_wavefront_radii`` combined with at most one
        diffraction interaction anywhere along the path is **not**
        currently supported (and raises at runtime, via
        :func:`equinox.error_if`): computing the diffraction point's
        edge-fixed radius of curvature would require tracking the
        wavefront's principal-axis orientation along the path, not just
        its two radii. For a single *isotropic* radius (or a planar
        wavefront), the existing behavior is unchanged: the UTD distance
        parameter and the post-diffraction (cylindrical) spreading factor
        both use the *cumulative* path length up to the diffraction
        point, plus ``tx_wavefront_radii``. This also only tracks the
        wavefront up to the first diffraction: a path with a diffraction
        bounce followed by further interactions would, in general, need
        to track the diffracted wavefront's curvature onward, since
        diffraction turns even an initially-spherical wavefront
        astigmatic; :func:`L_i<differt.em.L_i>` already accepts the
        general astigmatic case (``rho_1_i``, ``rho_2_i``, ``rho_e_i``)
        for whoever wants to extend :meth:`diffraction_matrix` that far.
        Sionna RT has no equivalent feature to compare against;
        ``tx_wavefront_radii``'s correctness instead rests on the
        geometric argument above (also checked against the equivalent of
        physically moving the transmitter back by
        ``tx_wavefront_radii``, see the test suite).

        When :attr:`tx_polarization` is set to an
        :class:`AbstractAntenna<differt.em.AbstractAntenna>` instance, its own
        :meth:`AbstractAntenna.wavefront_radii<differt.em.AbstractAntenna.wavefront_radii>`
        is used *instead of* :attr:`tx_wavefront_radii`.
    """

    supported_interaction_types: ClassVar[frozenset[InteractionType]] = frozenset({
        InteractionType.REFLECTION,
        InteractionType.DIFFRACTION,
        InteractionType.TRANSMISSION,
        InteractionType.SCATTERING,
    })
    """The interaction types handled by :meth:`transition_matrices`.

    Subclasses adding support for a new interaction type must add the
    corresponding member here, in addition to overriding the matching
    ``*_matrix`` method.
    """

    tx_polarization: Any = "V"
    """The transmitter antenna polarization or pattern.

    Either ``"V"``, ``"H"``, a Jones vector, or an
    :class:`AbstractAntenna<differt.em.AbstractAntenna>` (that provides ``.fields(...)``
    and ``.wavefront_radii(...)``).

    To model different antennas across a scene, pass a single antenna
    instance whose array fields carry their own batch dimension,
    broadcastable against the paths' batch dimensions -- either one
    antenna model shared by every transmitter, or one antenna model per
    transmitter (:class:`AbstractAntenna<differt.em.AbstractAntenna>` is a
    :class:`equinox.Module`, so this works out of the box).

    When this is an :class:`AbstractAntenna<differt.em.AbstractAntenna>` instance, its
    :meth:`AbstractAntenna.wavefront_radii<differt.em.AbstractAntenna.wavefront_radii>`
    is used *instead of* :attr:`tx_wavefront_radii` (which only serves as
    a fallback for a plain polarization string/vector, since those carry
    no wavefront-curvature information of their own); its
    :attr:`~BaseAntenna.frequency` is also used by
    :func:`compute_received_fields<differt.em.compute_received_fields>`
    when no ``frequency`` is passed explicitly.
    """
    rx_polarization: Any = "V"
    """The receiver antenna polarization or pattern. See :attr:`tx_polarization`."""
    radio_materials: Mapping[str, Material] | None = None
    """The mapping of material properties.

    Defaults to :data:`materials<differt.em._material.materials>` when
    left to :data:`None`.
    """
    tx_wavefront_radii: (
        Float[ArrayLike, "*#batch"]
        | tuple[Float[ArrayLike, "*#batch"], Float[ArrayLike, "*#batch"]]
        | None
    ) = 0.0
    r"""The radius (or radii) of curvature of the incident wavefront at the transmitter.

    For a non-planar (near-field) source. This is a distance, and ``0``
    and :data:`None` are its two opposite limits, *not* two ways of
    saying the same thing: ``0`` (the default) is the near-distance
    limit, an ideal point source located exactly at the transmitter,
    matching Sionna RT's implicit assumption; :data:`None` is the
    far-distance limit (:math:`\rho_0 \to \infty`), an ideal plane wave,
    e.g., a source far enough away that its curvature is negligible --
    see :class:`AbstractFarFieldAntenna<differt.em.AbstractFarFieldAntenna>`. Either of
    those, a single finite value (spherical wavefront), or a
    ``(rho_s, rho_p)`` tuple (astigmatic wavefront, with unequal
    principal radii along the s- and p-planes -- not supported together
    with a ``DIFFRACTION`` interaction) may be passed. Ignored when
    :attr:`tx_polarization` is an :class:`AbstractAntenna<differt.em.AbstractAntenna>`
    instance -- see :attr:`tx_polarization`.
    """

    @property
    def _radio_materials(self) -> Mapping[str, Material]:
        """:attr:`radio_materials`, falling back to :data:`materials<differt.em._material.materials>`."""
        return self.radio_materials if self.radio_materials is not None else materials

    def _resolve_tx_wavefront_radii(
        self,
        paths: TracedPaths,
    ) -> (
        Float[ArrayLike, "*#batch"]
        | tuple[Float[ArrayLike, "*#batch"], Float[ArrayLike, "*#batch"]]
        | tuple[
            Float[ArrayLike, "*#batch"],
            Float[ArrayLike, "*#batch 3"],
            Float[ArrayLike, "*#batch"],
            Float[ArrayLike, "*#batch 3"],
        ]
        | None
    ):
        """
        Resolve the effective wavefront radius (or radii) for the given paths.

        Uses ``self.tx_polarization.wavefront_radii(k_hat)`` (with
        ``k_hat`` the direction of each path's first segment) whenever
        :attr:`tx_polarization` is an :class:`AbstractAntenna<differt.em.AbstractAntenna>`
        (or, more precisely, provides a ``wavefront_radii`` method),
        falling back to :attr:`tx_wavefront_radii` otherwise.

        Returns:
            The value to pass to :func:`_wavefront_radii`.
        """
        if hasattr(self.tx_polarization, "wavefront_radii"):
            path_segments = jnp.diff(paths.vertices, axis=-2)
            k_hat = normalize(path_segments, keepdims=True)[0][..., 0, :]
            return self.tx_polarization.wavefront_radii(k_hat)
        return self.tx_wavefront_radii

    def reflection_matrix(
        self,
        paths: TracedPaths,
        mesh: Mesh,
        frequency: Float[ArrayLike, "*#batch"],
    ) -> Complex[Array, "*batch order 2 2"]:
        """
        Compute the per-bounce reflection transition matrix, for every bounce.

        The reflection coefficients are obtained from
        :func:`reflection_coefficients<differt.em.reflection_coefficients>`
        (or the thin-slab variant thereof, when the material has a
        finite thickness), then expressed in the local (theta, phi)
        spherical basis of the incoming and outgoing ray segments.

        Args:
            paths: The paths.
            mesh: The triangle mesh of the scene.
            frequency: The operating frequency (or frequencies) in Hz.

        Returns:
            One transition matrix per bounce.
        """
        frequency = jnp.asarray(frequency)
        k, k_in, obj_normals, n_r_val, thickness_val, cos_theta_i, wavelength = (
            _surface_interaction_geometry(paths, mesh, frequency, self._radio_materials)
        )
        k_out = k[..., 1:, :]

        (e_i_s, e_i_p), (e_r_s, e_r_p) = sp_directions(k_in, k_out, obj_normals)

        r_s, r_p = _get_reflection_coefficients(
            n_r_val, cos_theta_i, thickness_val, wavelength[..., None]
        )

        # If the material also has a nonzero 'scattering_coefficient' (S),
        # a fraction S^2 of the reflected power is diverted to diffuse
        # scattering (see 'scattering_matrix'); reduce the specular
        # amplitude accordingly to conserve energy between the two.
        obj_indices = jnp.clip(paths.objects[..., 1:-1], 0, mesh.num_triangles - 1)
        mat_indices = jnp.take(mesh.face_materials, obj_indices, axis=0)
        scattering_coefficient, _ = _scattering_properties(mesh, self._radio_materials)
        s_val = _take_material_property(scattering_coefficient, mat_indices)
        specular_factor = jnp.sqrt(1.0 - s_val**2)
        r_s, r_p = r_s * specular_factor, r_p * specular_factor

        theta_hat, phi_hat = _spherical_basis(k)
        theta_in, phi_in = theta_hat[..., :-1, :], phi_hat[..., :-1, :]
        theta_out, phi_out = theta_hat[..., 1:, :], phi_hat[..., 1:, :]

        in_rot = sp_rotation_matrix(theta_in, phi_in, e_i_s, e_i_p)
        out_rot = sp_rotation_matrix(e_r_s, e_r_p, theta_out, phi_out)

        zero = jnp.zeros_like(r_s)
        d_j = jnp.stack(
            [jnp.stack([r_s, zero], axis=-1), jnp.stack([zero, r_p], axis=-1)],
            axis=-2,
        )

        return jnp.matmul(out_rot, jnp.matmul(d_j, in_rot))

    def diffraction_matrix(
        self,
        paths: TracedPaths,
        mesh: Mesh,
        frequency: Float[ArrayLike, "*#batch"],
    ) -> Complex[Array, "*batch order 2 2"]:
        r"""
        Compute the per-bounce diffraction transition matrix, for every bounce.

        Implements the Uniform Theory of Diffraction (UTD), via
        :func:`diffraction_coefficients<differt.em.diffraction_coefficients>`,
        for a wedge formed by two adjacent mesh faces. The convention for a
        ``DIFFRACTION`` bounce is that :attr:`TracedPaths.objects
        <differt.geometry.TracedPaths.objects>` holds a flat *half-edge*
        index ``3 * triangle_index + local_edge_index`` (``local_edge_index``
        being 0, 1, or 2), matching Sionna RT's own wedge addressing and
        :attr:`Mesh.wedge_angles<differt.geometry.Mesh.wedge_angles>`'s
        ``(num_triangles, 3)`` shape — unlike ``REFLECTION``/``TRANSMISSION``,
        which index a plain *triangle*. This (rather than
        :attr:`Mesh.diffraction_edges<differt.geometry.Mesh.diffraction_edges>`,
        which deduplicates shared edges via a non-:func:`jax.jit`-compatible
        :func:`~jax.numpy.unique`) keeps this method :func:`jax.jit`-compatible.

        The wedge-face reflection coefficients (used for the shadow- and
        reflection-boundary terms) use the same finite-thickness slab model
        as :meth:`reflection_matrix`, matching Sionna RT's dielectric-wedge
        diffraction model.

        The transmitter's non-planar wavefront (:attr:`tx_wavefront_radii`,
        or the value returned by
        :meth:`AbstractAntenna.wavefront_radii<differt.em.AbstractAntenna.wavefront_radii>`
        when :attr:`tx_polarization` is an
        :class:`AbstractAntenna<differt.em.AbstractAntenna>`) only affects a
        ``DIFFRACTION`` bounce that is the *first* interaction along a
        path; see the note in :class:`GeometricFieldSolver`. :data:`None`
        (a planar wavefront, the :math:`\rho_0 \to \infty` limit) is
        supported here (unlike an astigmatic ``(rho_s, rho_p)`` tuple,
        which is not, and must not be used with a ``DIFFRACTION`` bounce
        -- see :meth:`spreading_factor`), using the well-known
        plane-wave-incidence distance parameter formula, rather than
        evaluating the general one at :math:`\rho_0 \to \infty` (which
        would give a 0/0 division).

        Args:
            paths: The paths.
            mesh: The triangle mesh of the scene.
            frequency: The operating frequency (or frequencies) in Hz.

        Returns:
            One transition matrix per bounce.

        Raises:
            ValueError: If the mesh does not contain face materials.
        """
        if mesh.face_materials is None:
            msg = "Mesh must contain face materials to compute diffraction."
            raise ValueError(msg)

        frequency = jnp.asarray(frequency)
        n0_he, nn_he, e_hat_he, primn_he = _wedge_static_geometry(mesh)

        # For a DIFFRACTION bounce, 'objects' holds a flat half-edge index
        # '3 * triangle_index + local_edge_index' (0, 1, or 2), rather than
        # a plain triangle index as for REFLECTION/TRANSMISSION.
        half_edge_idx = paths.objects[..., 1:-1]
        prim0_idx, local_edge_idx = (
            half_edge_idx // 3,
            half_edge_idx % 3,
        )

        n0 = n0_he[prim0_idx, local_edge_idx]
        nn = nn_he[prim0_idx, local_edge_idx]
        e_hat = e_hat_he[prim0_idx, local_edge_idx]
        prim0 = prim0_idx
        primn = primn_he[prim0_idx, local_edge_idx]
        wedge_n = mesh.wedge_angles[prim0_idx, local_edge_idx]

        path_segments = jnp.diff(paths.vertices, axis=-2)
        k, s = normalize(path_segments, keepdims=True)
        k_in = k[..., :-1, :]
        k_out = k[..., 1:, :]
        s_prime = s[..., :-1, 0]
        s_out = s[..., 1:, 0]

        # Sionna RT orients the 0-/n-face labeling per bounce, based on the
        # incident ray's propagation direction, so that the 0-face is the
        # one actually illuminated by the incident ray.
        swap = jnp.sum(k_in * n0, axis=-1) > 0.0
        n0, nn = jnp.where(swap[..., None], nn, n0), jnp.where(swap[..., None], n0, nn)
        e_hat = jnp.where(swap[..., None], -e_hat, e_hat)
        prim0, primn = jnp.where(swap, primn, prim0), jnp.where(swap, prim0, primn)

        t0_hat, _ = normalize(jnp.cross(n0, e_hat))

        # Non-diffracting (e.g., REFLECTION/TRANSMISSION) bounces still flow
        # through this method (their result is discarded later, based on
        # 'interaction_types'), but may hit degenerate geometry (e.g., a ray
        # parallel to a placeholder edge); clip all inverse-trigonometric
        # inputs to avoid ever producing a NaN, which would otherwise poison
        # the 'jnp.where'-based combination in 'transition_matrices'.
        ki_dot_e = jnp.sum(k_in * e_hat, axis=-1, keepdims=True)
        ki_proj, _ = normalize(k_in - ki_dot_e * e_hat)
        ko_dot_e = jnp.sum(k_out * e_hat, axis=-1, keepdims=True)
        ko_proj, _ = normalize(k_out - ko_dot_e * e_hat)

        phi_prime = jnp.pi - jnp.arccos(
            jnp.clip(-jnp.sum(ki_proj * t0_hat, axis=-1), -1.0, 1.0)
        )
        phi_prime = phi_prime * -jnp.sign(-jnp.sum(ki_proj * n0, axis=-1))
        phi_prime = phi_prime + jnp.pi

        phi = jnp.pi - jnp.arccos(
            jnp.clip(jnp.sum(ko_proj * t0_hat, axis=-1), -1.0, 1.0)
        )
        phi = phi * -jnp.sign(jnp.sum(ko_proj * n0, axis=-1))
        phi = phi + jnp.pi

        cos_beta_0 = jnp.clip(jnp.abs(jnp.sum(k_in * e_hat, axis=-1)), 0.0, 1.0)
        sin_beta_0 = jnp.sqrt(1.0 - cos_beta_0**2)

        # The incident wavefront's radius of curvature at the diffraction
        # point is 's_prime' away from an ideal point source at the
        # transmitter; only add the resolved wavefront radius when this
        # bounce is the *first* interaction (order index 0), since it is
        # only there that 's_prime' is the distance from the transmitter
        # itself.
        radii = _wavefront_radii(self._resolve_tx_wavefront_radii(paths))
        is_first_bounce = jnp.arange(s_prime.shape[-1]) == 0

        if radii is None:
            # A planar wavefront has no associated point-source distance to
            # add; for a first-bounce diffraction, this is exactly the
            # well-known plane-wave-incidence formula (the radii-based
            # formula below is not simply evaluated at 'rho_i -> inf', which
            # would be a 0/0 (NaN) division).
            L_planar = s_out * sin_beta_0**2  # ruff:ignore[non-lowercase-variable-in-function]
            L_other = safe_divide(s_prime * s_out, s_prime + s_out) * sin_beta_0**2  # ruff:ignore[non-lowercase-variable-in-function]
            L = jnp.where(is_first_bounce, L_planar, L_other)  # ruff:ignore[non-lowercase-variable-in-function]
        else:
            rho_s, rho_p = radii
            # This method runs on every bounce (its result is discarded
            # later, based on 'interaction_types'), so only complain when an
            # astigmatic radius is combined with an *actual* DIFFRACTION
            # bounce.
            is_actual_diffraction = (
                paths.interaction_types == InteractionType.DIFFRACTION
            )
            rho_s = eqx.error_if(
                rho_s,
                jnp.any(is_actual_diffraction) & jnp.any(rho_s != rho_p),
                "An astigmatic 'tx_wavefront_radii' (a '(rho_s, rho_p)' tuple "
                "with 'rho_s != rho_p') is not currently supported for a "
                "DIFFRACTION interaction; pass a single (spherical) value "
                "instead.",
            )
            rho_i = s_prime + jnp.where(is_first_bounce, rho_s[..., None], 0.0)

            L = (  # ruff:ignore[non-lowercase-variable-in-function]
                safe_divide(rho_i * s_out, rho_i + s_out) * sin_beta_0**2
            )

        n_complex, thickness = _material_arrays(mesh, self._radio_materials, frequency)
        mat0_idx = jnp.take(mesh.face_materials, prim0, axis=0)
        matn_idx = jnp.take(mesh.face_materials, primn, axis=0)
        n_r_o = _take_material_property(n_complex, mat0_idx)
        n_r_n = _take_material_property(n_complex, matn_idx)
        # A material with no explicit thickness uses the '-1' sentinel
        # (meaning "infinite half-space", see '_material_arrays'); passed
        # through as-is, 'diffraction_coefficients' resolves this sentinel
        # per-element the same way '_get_reflection_coefficients' does for
        # plain REFLECTION.
        d_o = jnp.take(thickness, mat0_idx, axis=0)
        d_n = jnp.take(thickness, matn_idx, axis=0)

        wavenumber = jnp.broadcast_to(2.0 * jnp.pi * frequency / c, wedge_n.shape)
        D_s, D_h = diffraction_coefficients(  # ruff:ignore[non-lowercase-variable-in-function]
            wavenumber,
            wedge_n,
            phi_prime,
            phi,
            L,
            sin_beta_0=sin_beta_0,
            n_r_o=n_r_o,
            n_r_n=n_r_n,
            d_o=d_o,
            d_n=d_n,
        )

        phi_hat_prime, _ = normalize(jnp.cross(k_in, e_hat))
        tau_hat_prime, _ = normalize(jnp.cross(phi_hat_prime, k_in))
        phi_hat_d, _ = normalize(jnp.cross(k_out, e_hat))
        phi_hat_d = -phi_hat_d
        tau_hat_d, _ = normalize(jnp.cross(phi_hat_d, k_out))

        theta_hat, phi_hat_sph = _spherical_basis(k)
        theta_in, phi_in = theta_hat[..., :-1, :], phi_hat_sph[..., :-1, :]
        theta_out, phi_out = theta_hat[..., 1:, :], phi_hat_sph[..., 1:, :]

        in_rot = sp_rotation_matrix(theta_in, phi_in, phi_hat_prime, tau_hat_prime)
        out_rot = sp_rotation_matrix(phi_hat_d, tau_hat_d, theta_out, phi_out)

        zero = jnp.zeros_like(D_s)
        d_j = jnp.stack(
            [jnp.stack([D_s, zero], axis=-1), jnp.stack([zero, D_h], axis=-1)],
            axis=-2,
        )

        return jnp.matmul(out_rot, jnp.matmul(d_j, in_rot))

    def scattering_matrix(
        self,
        paths: TracedPaths,
        mesh: Mesh,
        frequency: Float[ArrayLike, "*#batch"],
    ) -> Complex[Array, "*batch order 2 2"]:
        r"""
        Compute the per-bounce diffuse scattering transition matrix, for every bounce.

        Implements a Lambertian rough-surface scattering model, following
        Degli-Esposti's directive model (as also used by Sionna RT's
        ``RadioMaterial``), adapted to DiffeRT's exact/deterministic path
        model:

        .. warning::

            Sionna RT's scattering model is defined in terms of a Monte
            Carlo ray-tube solid angle :math:`\Omega`, which has no
            equivalent for a single, deterministic point-to-point path.
            This method instead uses the solid angle subtended by the
            scattering triangle, as seen from the next path vertex,
            :math:`\mathrm{d}A / s^2`, as a geometrically-motivated
            substitute (:math:`\mathrm{d}A` being the triangle's area and
            :math:`s` the distance to the next vertex). This is a
            physically-motivated but distinct adaptation: because Sionna
            RT's model is stochastic, this method should **not** be
            expected to numerically match its output for a single path,
            only in a statistical (many-samples) sense.

        Given the specular reflection coefficients :math:`r_s, r_p`
        (:meth:`reflection_matrix`, ignoring the scattering-coefficient
        energy reduction) and the incident and scattered directions
        :math:`\hat{k}_i, \hat{k}_o`, the per-polarization scattered
        amplitude is:

        .. math::
            a_{s,p} = S \sqrt{f_s(\hat{k}_i, \hat{k}_o, \hat{n}) \frac{\mathrm{d}A}{s^2}} \, |r_{s,p}|,

        where :math:`S` is
        :attr:`Material.scattering_coefficient<differt.em._material.Material.scattering_coefficient>`
        and :math:`f_s` is the material's
        :attr:`Material.scattering_pattern<differt.em._material.Material.scattering_pattern>`
        (normalized so that its integral over the hemisphere is 1),
        which defaults to
        :class:`LambertianPattern<differt.em._material.LambertianPattern>`,
        :math:`f_s(\hat{k}_i, \hat{k}_o, \hat{n}) = \max(\hat{n}\cdot\hat{k}_o, 0) / \pi`.
        A custom (e.g., directive) pattern can be set per-material by
        subclassing
        :class:`AbstractScatteringPattern<differt.em._material.AbstractScatteringPattern>`.
        A final rotation mixes the s and p channels according to
        :attr:`Material.xpd_coefficient<differt.em._material.Material.xpd_coefficient>`.

        Args:
            paths: The paths.
            mesh: The triangle mesh of the scene.
            frequency: The operating frequency (or frequencies) in Hz.

        Returns:
            One transition matrix per bounce.
        """
        frequency = jnp.asarray(frequency)
        k, k_in, obj_normals, n_r_val, thickness_val, cos_theta_i, wavelength = (
            _surface_interaction_geometry(paths, mesh, frequency, self._radio_materials)
        )
        k_out = k[..., 1:, :]

        r_s, r_p = _get_reflection_coefficients(
            n_r_val, cos_theta_i, thickness_val, wavelength[..., None]
        )
        gamma_s, gamma_p = jnp.abs(r_s), jnp.abs(r_p)

        obj_indices = jnp.clip(paths.objects[..., 1:-1], 0, mesh.num_triangles - 1)
        mat_indices = jnp.take(mesh.face_materials, obj_indices, axis=0)
        scattering_coefficient, xpd_coefficient = _scattering_properties(
            mesh, self._radio_materials
        )
        s_val = _take_material_property(scattering_coefficient, mat_indices)
        xpd_val = _take_material_property(xpd_coefficient, mat_indices)

        triangle_vertices = jnp.take(mesh.triangle_vertices, obj_indices, axis=0)
        edge_1 = triangle_vertices[..., 1, :] - triangle_vertices[..., 0, :]
        edge_2 = triangle_vertices[..., 2, :] - triangle_vertices[..., 0, :]
        triangle_area = 0.5 * jnp.linalg.norm(jnp.cross(edge_1, edge_2), axis=-1)

        path_segments = jnp.diff(paths.vertices, axis=-2)
        _, s = normalize(path_segments, keepdims=True)
        s_out = s[..., 1:, 0]

        f_s = _scattering_pattern_values(
            mesh, self._radio_materials, k_in, k_out, obj_normals, mat_indices
        )

        solid_angle = safe_divide(triangle_area, s_out**2)
        amplitude = s_val * jnp.sqrt(f_s * solid_angle)
        a_s, a_p = amplitude * gamma_s, amplitude * gamma_p

        (e_i_s, e_i_p), (e_r_s, e_r_p) = sp_directions(k_in, k_out, obj_normals)

        theta_hat, phi_hat = _spherical_basis(k)
        theta_in, phi_in = theta_hat[..., :-1, :], phi_hat[..., :-1, :]
        theta_out, phi_out = theta_hat[..., 1:, :], phi_hat[..., 1:, :]

        in_rot = sp_rotation_matrix(theta_in, phi_in, e_i_s, e_i_p)
        out_rot = sp_rotation_matrix(e_r_s, e_r_p, theta_out, phi_out)

        # Real-valued (no phase shift beyond the overall path phase), but
        # cast to complex for dtype-consistency with the other interaction
        # types' matrices, which 'transition_matrices' combines via
        # 'jnp.where'.
        dtype = jnp.result_type(paths.vertices)
        cdtype = jnp.complex128 if dtype == jnp.float64 else jnp.complex64
        zero = jnp.zeros_like(a_s, dtype=cdtype)
        a_s, a_p = a_s.astype(cdtype), a_p.astype(cdtype)
        d_j = jnp.stack(
            [jnp.stack([a_s, zero], axis=-1), jnp.stack([zero, a_p], axis=-1)],
            axis=-2,
        )

        theta_x = jnp.arcsin(jnp.sqrt(xpd_val))
        cos_x, sin_x = jnp.cos(theta_x), jnp.sin(theta_x)
        j_xpd = jnp.stack(
            [jnp.stack([cos_x, -sin_x], axis=-1), jnp.stack([sin_x, cos_x], axis=-1)],
            axis=-2,
        ).astype(cdtype)

        return jnp.matmul(out_rot, jnp.matmul(j_xpd, jnp.matmul(d_j, in_rot)))

    def transmission_matrix(
        self,
        paths: TracedPaths,
        mesh: Mesh,
        frequency: Float[ArrayLike, "*#batch"],
    ) -> Complex[Array, "*batch order 2 2"]:
        """
        Compute the per-bounce transmission transition matrix, for every bounce.

        This is a **thin-surface** transmission model: the transmitting
        object is treated as a single dielectric slab whose thickness is
        read from :attr:`Material.thickness<differt.em._material.Material.thickness>`
        (accounting for multiple internal reflections in closed form, per
        ITU-R P.2040-3 eq. 43b/44), evaluated at the incidence angle and
        combined coherently. This matches Sionna RT's
        ``RadioMaterial``/``ITURadioMaterial`` transmission model, and, as
        in Sionna RT, the ray direction is **not** bent by refraction (the
        wall is assumed thin enough that the geometric deflection is
        negligible); only the field amplitude/phase are affected.

        This is a deliberate simplification, not an architectural
        constraint: a **volumetric** transmission model, where the
        effective thickness is instead derived from where a ray enters and
        exits a solid object's geometry (rather than from a fixed
        per-material property), can be added later as a sibling
        implementation, by overriding this method in a subclass.

        Args:
            paths: The paths.
            mesh: The triangle mesh of the scene.
            frequency: The operating frequency (or frequencies) in Hz.

        Returns:
            One transition matrix per bounce.
        """
        frequency = jnp.asarray(frequency)
        k, k_in, obj_normals, n_r_val, thickness_val, cos_theta_i, wavelength = (
            _surface_interaction_geometry(paths, mesh, frequency, self._radio_materials)
        )

        # This method may run on bounces that are not actually a
        # TRANSMISSION interaction (its result is discarded later, based on
        # 'interaction_types'); only complain about a missing thickness for
        # bounces that are actually used as a transmission.
        is_transmission = paths.interaction_types == InteractionType.TRANSMISSION
        thickness_val = eqx.error_if(
            thickness_val,
            jnp.any((thickness_val < 0.0) & is_transmission),
            "Materials used in a TRANSMISSION interaction must have a finite "
            "'thickness' set (e.g., Material(..., thickness=0.1)); materials "
            "default to an infinite half-space (thickness=None), which is "
            "meaningless for transmission.",
        )
        thickness_val = jnp.maximum(thickness_val, 0.0)

        # Transmission does not bend the ray: the "outgoing" direction is the
        # incident one, so the local s/p basis is the same on both sides.
        (e_i_s, e_i_p), _ = sp_directions(k_in, k_in, obj_normals)

        _, (t_s, t_p) = slab_coefficients(
            n_r_val, cos_theta_i, thickness_val, wavelength[..., None]
        )

        theta_hat, phi_hat = _spherical_basis(k)
        theta_in, phi_in = theta_hat[..., :-1, :], phi_hat[..., :-1, :]

        in_rot = sp_rotation_matrix(theta_in, phi_in, e_i_s, e_i_p)
        out_rot = sp_rotation_matrix(e_i_s, e_i_p, theta_in, phi_in)

        zero = jnp.zeros_like(t_s)
        d_j = jnp.stack(
            [jnp.stack([t_s, zero], axis=-1), jnp.stack([zero, t_p], axis=-1)],
            axis=-2,
        )

        return jnp.matmul(out_rot, jnp.matmul(d_j, in_rot))

    def ris_matrix(
        self,
        paths: TracedPaths,
        mesh: Mesh,
        frequency: Float[ArrayLike, "*#batch"],
    ) -> Complex[Array, "*batch order 2 2"]:
        """
        Compute the per-bounce RIS (Reconfigurable Intelligent Surface) transition matrix.

        .. warning::

            Not implemented yet. Subclasses can override this method to provide custom RIS physics.

        Args:
            paths: The paths.
            mesh: The triangle mesh of the scene.
            frequency: The operating frequency (or frequencies) in Hz.

        Raises:
            NotImplementedError: Unconditionally, as RIS modeling is reserved for future implementation.
        """
        msg = "RIS matrix computation is not implemented yet. Override 'ris_matrix' in a subclass."
        raise NotImplementedError(msg)

    def transition_matrices(
        self,
        paths: TracedPaths,
        mesh: Mesh,
        frequency: Float[ArrayLike, "*#batch"],
    ) -> Complex[Array, "*batch order 2 2"]:
        """
        Compute one transition matrix per bounce, dispatching on the bounce's interaction type.

        Only interaction types listed in :attr:`supported_interaction_types`
        are ever dispatched to (a plain Python, non-traced, check), so
        overriding one of the ``*_matrix`` methods without adding the type
        here has no effect. Any bounce whose interaction type is not covered
        raises at runtime (works under :func:`jax.jit` too).

        Args:
            paths: The paths.
            mesh: The triangle mesh of the scene.
            frequency: The operating frequency (or frequencies) in Hz.

        Returns:
            One transition matrix per bounce.
        """
        interaction_types = paths.interaction_types
        hooks = {
            InteractionType.REFLECTION: self.reflection_matrix,
            InteractionType.DIFFRACTION: self.diffraction_matrix,
            InteractionType.SCATTERING: self.scattering_matrix,
            InteractionType.TRANSMISSION: self.transmission_matrix,
            InteractionType.RIS: self.ris_matrix,
        }

        dtype = jnp.result_type(paths.vertices)
        cdtype = jnp.complex128 if dtype == jnp.float64 else jnp.complex64
        mat = jnp.broadcast_to(
            jnp.eye(2, dtype=cdtype), (*interaction_types.shape, 2, 2)
        )
        covered = jnp.zeros(interaction_types.shape, dtype=bool)

        for interaction_type in self.supported_interaction_types:
            type_mat = hooks[interaction_type](paths, mesh, frequency)
            is_type = interaction_types == interaction_type
            mat = jnp.where(is_type[..., None, None], type_mat, mat)
            covered = covered | is_type

        is_padding = interaction_types == -1
        unsupported = covered | is_padding
        return eqx.error_if(
            mat,
            ~jnp.all(unsupported),
            "TracedPaths contains an interaction type that this GeometricFieldSolver "
            "does not support (i.e., not listed in 'supported_interaction_types'). "
            "Override the matching '*_matrix' method and add the corresponding "
            "InteractionType to 'supported_interaction_types' in a subclass.",
        )

    def compute_fields(
        self,
        paths: TracedPaths,
        mesh: Mesh,
        frequency: Float[ArrayLike, "*#batch"],
    ) -> Complex[Array, "*batch"]:
        """
        Compute the received complex fields for each path.

        Reads :attr:`tx_polarization`, :attr:`rx_polarization`,
        :attr:`radio_materials`, and :attr:`tx_wavefront_radii` from
        ``self`` -- see :class:`GeometricFieldSolver`.

        Args:
            paths: The paths.
            mesh: The triangle mesh of the scene.
            frequency: The operating frequency (or frequencies) in Hz.

                Can be an array broadcastable against ``paths``' batch
                dimensions, e.g., to assign a different frequency to
                different transmitters.

        Returns:
            The received complex fields of shape ``*batch``.
        """
        tx_polarization = self.tx_polarization
        rx_polarization = self.rx_polarization

        frequency = jnp.asarray(frequency)
        wavelength = c / frequency

        path_segments = jnp.diff(paths.vertices, axis=-2)
        k, s = normalize(path_segments, keepdims=True)

        theta_hat_arr, phi_hat_arr = _spherical_basis(k)

        theta_hat_0 = theta_hat_arr[..., 0, :]
        phi_hat_0 = phi_hat_arr[..., 0, :]

        if hasattr(tx_polarization, "fields"):
            T = paths.vertices[..., 0, :]
            r_hat = k[..., 0, :]
            e_init, _ = tx_polarization.fields(T + r_hat)
            e_dir = e_init * jnp.exp(1j * tx_polarization.wavenumber)
            e_theta = jnp.sum(e_dir * theta_hat_0, axis=-1)
            e_phi = jnp.sum(e_dir * phi_hat_0, axis=-1)
            e_field = jnp.stack([e_theta, e_phi], axis=-1)
        elif hasattr(tx_polarization, "polarization_vectors"):
            T = paths.vertices[..., 0, :]
            r_hat = k[..., 0, :]
            s_vec, p_vec = tx_polarization.polarization_vectors(T + r_hat)
            e_theta = jnp.sum(s_vec * theta_hat_0, axis=-1) + jnp.sum(
                p_vec * theta_hat_0, axis=-1
            )
            e_phi = jnp.sum(s_vec * phi_hat_0, axis=-1) + jnp.sum(
                p_vec * phi_hat_0, axis=-1
            )
            e_field = jnp.stack([e_theta, e_phi], axis=-1).astype(complex)
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

        e_field_vec = e_field[..., None]

        if paths.order > 0:
            j_mat = self.transition_matrices(paths, mesh, frequency)

            j_list = [j_mat[..., j, :, :] for j in range(paths.order)]
            j_total = functools.reduce(lambda x, y: jnp.matmul(y, x), j_list)
            e_field_vec = jnp.matmul(j_total, e_field_vec)
            e_field = e_field_vec[..., 0]

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
        elif hasattr(rx_polarization, "polarization_vectors"):
            r = paths.vertices[..., -1, :]
            k_last = k[..., -1, :]
            s_vec, p_vec = rx_polarization.polarization_vectors(r - k_last)
            u_theta = jnp.sum(s_vec * theta_hat_last, axis=-1) + jnp.sum(
                p_vec * theta_hat_last, axis=-1
            )
            u_phi = jnp.sum(s_vec * phi_hat_last, axis=-1) + jnp.sum(
                p_vec * phi_hat_last, axis=-1
            )
            u = jnp.stack([u_theta, u_phi], axis=-1).astype(complex)
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

        s_tot = s.sum(axis=-2)[..., 0]
        spreading_factor = self.spreading_factor(paths)
        # The virtual source is 'tx_wavefront_radii' further away than the
        # transmitter itself, along the first segment's direction, so the
        # wave accumulates that much extra propagation phase too (same
        # total distance that already enters 'spreading_factor' above). For
        # an astigmatic source, the two principal radii generally do not
        # correspond to a single virtual point source, so their mean is
        # used as a reasonable single extra-distance value for phase
        # purposes; this reduces to the exact spherical-source result when
        # both radii are equal (the default, and the only case combined
        # with a DIFFRACTION interaction). A planar wavefront has no
        # associated point-source distance at all, so no extra phase is
        # added in that case.
        radii = _wavefront_radii(self._resolve_tx_wavefront_radii(paths))
        extra_distance = 0.5 * (radii[0] + radii[1]) if radii is not None else 0.0
        phase_val = -2.0 * jnp.pi * frequency * (s_tot + extra_distance) / c
        phase_shift = jax.lax.complex(jnp.cos(phase_val), jnp.sin(phase_val))

        a_r = a_r * spreading_factor * phase_shift
        a = a_r * (wavelength / (4 * jnp.pi))

        return a * paths.mask

    def spreading_factor(self, paths: TracedPaths) -> Float[Array, "*batch"]:
        r"""
        Compute the wavefront spreading factor for each path.

        Reads :attr:`tx_wavefront_radii` (or the value returned by
        :meth:`AbstractAntenna.wavefront_radii<differt.em.AbstractAntenna.wavefront_radii>`
        when :attr:`tx_polarization` is an
        :class:`AbstractAntenna<differt.em.AbstractAntenna>`) from ``self``.

        For a path with no diffraction interaction, this is the general
        astigmatic-ray-tube spreading factor,
        :math:`1/\sqrt{(\rho_s+L)(\rho_p+L)}`, where :math:`L` is the
        total path length and :math:`\rho_s`, :math:`\rho_p` are the
        wavefront's two principal radii of curvature at the transmitter
        (plus :attr:`tx_wavefront_radii`, for a non-planar source); this is
        exact for any chain of flat-mirror reflections and/or
        transmissions, regardless of how many interactions the path has,
        since neither changes a wavefront's principal radii (only a flat
        mirror's image-method folds the ray direction). If
        :attr:`tx_wavefront_radii` is :data:`None` (a planar wavefront), this
        is instead exactly ``1`` (a plane wave does not spread at all).

        For a path containing a diffraction interaction, the wavefront
        becomes cylindrical past the diffraction point, and the spreading
        factor becomes :math:`1/\sqrt{\rho^i s(\rho^i+s)}`, where
        :math:`\rho^i` is the incident wavefront's radius of curvature at
        the diffraction point (the cumulative path length before it, plus
        :attr:`tx_wavefront_radii` if it is the first interaction) and
        :math:`s` is the path length after it, matching Sionna RT's model
        when :attr:`tx_wavefront_radii` is left at its default of ``0``. For
        a planar wavefront, this is instead :math:`1/\sqrt{s}`, the
        classic cylindrical-spreading law for an edge illuminated by a
        plane wave (:math:`\rho^i \to \infty` independently of :math:`s`,
        unlike the amplitude, which would incorrectly vanish if this
        limit were taken naively).

        Only a single diffraction interaction per path is currently
        supported; override this method to support multiple diffractions
        (or other custom wavefront models) per path. A genuinely
        astigmatic source (unequal principal radii) combined with a
        diffraction interaction is not currently supported either --
        computing the diffraction point's edge-fixed radius of curvature
        would require tracking the wavefront's principal-axis orientation
        along the path, not just its two radii -- and raises at runtime.

        Args:
            paths: The paths.

        Returns:
            The spreading factor for each path.
        """
        path_segments = jnp.diff(paths.vertices, axis=-2)
        _, s = normalize(path_segments, keepdims=True)
        s = s[..., 0]
        s_tot = s.sum(axis=-1)
        radii = _wavefront_radii(self._resolve_tx_wavefront_radii(paths))

        if radii is None:
            if paths.order == 0:
                return jnp.ones_like(s_tot)

            is_diffraction = paths.interaction_types == InteractionType.DIFFRACTION
            has_diffraction = jnp.any(is_diffraction, axis=-1)

            diffraction_index = jnp.argmax(is_diffraction, axis=-1)
            segment_index = jnp.arange(s.shape[-1])
            is_before = segment_index <= diffraction_index[..., None]
            s_after = jnp.sum(jnp.where(~is_before, s, 0.0), axis=-1)

            spreading_diffraction = safe_divide(1.0, jnp.sqrt(s_after))
            spreading_go = jnp.ones_like(s_tot)

            return jnp.where(has_diffraction, spreading_diffraction, spreading_go)

        rho_s, rho_p = radii

        if paths.order == 0:
            return safe_divide(1.0, jnp.sqrt((s_tot + rho_s) * (s_tot + rho_p)))

        is_diffraction = paths.interaction_types == InteractionType.DIFFRACTION
        has_diffraction = jnp.any(is_diffraction, axis=-1)

        is_astigmatic = jnp.any(rho_s != rho_p)
        rho_s = eqx.error_if(
            rho_s,
            jnp.any(has_diffraction) & is_astigmatic,
            "An astigmatic 'tx_wavefront_radii' (a '(rho_s, rho_p)' tuple with "
            "'rho_s != rho_p') is not currently supported for a path containing a "
            "DIFFRACTION interaction; pass a single (spherical) value instead.",
        )
        # In the (only currently supported) case reaching this point without
        # erroring, either 'rho_s == rho_p' (isotropic) for every path with a
        # diffraction interaction, or no path has one; either way, 'rho_s'
        # alone is the correct radius to use below.
        diffraction_index = jnp.argmax(is_diffraction, axis=-1)
        segment_index = jnp.arange(s.shape[-1])
        is_before = segment_index <= diffraction_index[..., None]
        s_prime = jnp.sum(jnp.where(is_before, s, 0.0), axis=-1) + rho_s
        s_after = jnp.sum(jnp.where(~is_before, s, 0.0), axis=-1)

        spreading_diffraction = safe_divide(
            1.0, jnp.sqrt(s_prime * s_after * (s_prime + s_after))
        )
        spreading_go = safe_divide(1.0, jnp.sqrt((s_tot + rho_s) * (s_tot + rho_p)))

        return jnp.where(has_diffraction, spreading_diffraction, spreading_go)
