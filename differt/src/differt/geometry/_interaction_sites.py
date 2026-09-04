"""Unified interaction-site index space, used to drive non-specular path tracing."""

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int

from ._mesh import Mesh

if TYPE_CHECKING:
    from differt.em import InteractionType

# Canonical order in which interaction types are laid out along the 'kind'/
# 'primitive' axis of 'InteractionSites'; arbitrary but fixed, so that
# 'build_interaction_sites' is deterministic for a given 'allowed_interactions'.
_SURFACE_INTERACTION_TYPES_ORDER = ("REFLECTION", "SCATTERING", "TRANSMISSION")


class InteractionSites(eqx.Module):
    """
    Flat, unified index space of everything a ray can interact with.

    Every entry maps a single integer "site" index (as enumerated by
    :class:`~differt_core.geometry.CompleteGraph`/:class:`~differt_core.geometry.DiGraph`,
    exactly as :attr:`Mesh.num_primitives<differt.geometry.Mesh.num_primitives>`
    triangle indices are today) to an
    :class:`InteractionType<differt.em.InteractionType>` and the primitive it
    refers to: a (quad-aware) triangle index for
    :attr:`REFLECTION<differt.em.InteractionType.REFLECTION>`,
    :attr:`SCATTERING<differt.em.InteractionType.SCATTERING>`, and
    :attr:`TRANSMISSION<differt.em.InteractionType.TRANSMISSION>`; a flat
    half-edge index ``3 * triangle_index + local_edge_index`` for
    :attr:`DIFFRACTION<differt.em.InteractionType.DIFFRACTION>`, matching
    :attr:`Mesh.wedge_angles<differt.geometry.Mesh.wedge_angles>`'s own
    half-edge addressing.
    """

    kind: Int[Array, " num_sites"]
    """The :class:`InteractionType<differt.em.InteractionType>` of each site."""
    primitive: Int[Array, " num_sites"]
    """The primitive (triangle or half-edge) index of each site."""


def build_interaction_sites(
    mesh: Mesh, allowed_interactions: "frozenset[InteractionType]"
) -> InteractionSites:
    """
    Build the flat interaction-site universe for a mesh and a set of allowed interactions.

    ``REFLECTION``, ``SCATTERING``, and ``TRANSMISSION`` each contribute one
    site per :attr:`Mesh.num_primitives<differt.geometry.Mesh.num_primitives>`
    (they all bounce, geometrically, off the same primitive); ``DIFFRACTION``
    contributes one site per half-edge slot (``3 * mesh.num_triangles``,
    matching :attr:`Mesh.wedge_angles<differt.geometry.Mesh.wedge_angles>`),
    most of which are not actual diffraction edges — see
    :func:`interaction_sites_valid_mask` to exclude those before enumerating
    paths.

    Args:
        mesh: The scene mesh.
        allowed_interactions: The set of interaction types to build sites for.

    Returns:
        The interaction sites.

    Raises:
        ValueError: If ``allowed_interactions`` is empty, or contains an
            interaction type that is not (yet) supported for path generation
            (currently, only ``REFLECTION``, ``SCATTERING``,
            ``TRANSMISSION``, and ``DIFFRACTION`` are).
    """
    from differt.em import InteractionType  # ruff: ignore[import-outside-top-level]

    supported = {
        InteractionType.REFLECTION,
        InteractionType.SCATTERING,
        InteractionType.TRANSMISSION,
        InteractionType.DIFFRACTION,
    }
    if not allowed_interactions:
        msg = "'allowed_interactions' must contain at least one InteractionType."
        raise ValueError(msg)
    if unsupported := allowed_interactions - supported:
        msg = (
            f"Unsupported interaction type(s) {sorted(unsupported)} in "
            "'allowed_interactions': path generation currently only supports "
            f"{sorted(supported)}."
        )
        raise ValueError(msg)

    kinds = []
    primitives = []

    num_primitives = mesh.num_primitives
    # 'primitive' must already be in the same (quad-aware) convention as
    # 'TracedPaths.objects': the first-triangle index of each quad (i.e.,
    # an even triangle index) when 'assume_quads' is set, matching a plain
    # triangle index otherwise.
    surface_primitives = jnp.arange(num_primitives, dtype=jnp.int32)
    if mesh.assume_quads:
        surface_primitives = 2 * surface_primitives
    for name in _SURFACE_INTERACTION_TYPES_ORDER:
        interaction_type = getattr(InteractionType, name)
        if interaction_type in allowed_interactions:
            kinds.append(jnp.full((num_primitives,), interaction_type, dtype=jnp.int32))
            primitives.append(surface_primitives)

    if InteractionType.DIFFRACTION in allowed_interactions:
        num_half_edges = mesh.num_triangles * 3
        kinds.append(
            jnp.full((num_half_edges,), InteractionType.DIFFRACTION, dtype=jnp.int32)
        )
        primitives.append(jnp.arange(num_half_edges, dtype=jnp.int32))

    return InteractionSites(
        kind=jnp.concatenate(kinds),
        primitive=jnp.concatenate(primitives),
    )


def interaction_sites_valid_mask(
    mesh: Mesh, sites: InteractionSites
) -> Bool[Array, " num_sites"]:
    """
    Identify sites that are structurally valid (as opposed to placeholder half-edge slots).

    ``REFLECTION``, ``SCATTERING``, and ``TRANSMISSION`` sites are always
    valid (any masking by :attr:`Mesh.mask<differt.geometry.Mesh.mask>` is
    applied later, exactly as for today's reflection-only candidates).
    ``DIFFRACTION`` sites are valid only where
    :attr:`Mesh.diffraction_edges_mask<differt.geometry.Mesh.diffraction_edges_mask>`
    is set, i.e., where the half-edge is an actual (non-boundary,
    non-coplanar) diffraction edge; this already accounts for
    :attr:`Mesh.mask<differt.geometry.Mesh.mask>`, if set.

    Args:
        mesh: The scene mesh that ``sites`` was built from.
        sites: The interaction sites, see :func:`build_interaction_sites`.

    Returns:
        A boolean mask, :data:`True` for structurally valid sites.
    """
    from differt.em import InteractionType  # ruff: ignore[import-outside-top-level]

    is_diffraction = sites.kind == InteractionType.DIFFRACTION
    safe_primitive = jnp.where(is_diffraction, sites.primitive, 0)
    diffraction_valid = mesh.diffraction_edges_mask.ravel()[safe_primitive]
    return jnp.where(is_diffraction, diffraction_valid, True)


def interaction_sites_mesh_mask(
    mesh: Mesh, sites: InteractionSites
) -> Bool[Array, " num_sites"]:
    """
    Identify sites whose underlying primitive is active per :attr:`Mesh.mask<differt.geometry.Mesh.mask>`.

    ``DIFFRACTION`` sites are always considered active here: masking is
    already accounted for by :func:`interaction_sites_valid_mask` (via
    :attr:`Mesh.diffraction_edges_mask<differt.geometry.Mesh.diffraction_edges_mask>`,
    which excludes half-edges next to an inactive triangle). ``REFLECTION``,
    ``SCATTERING``, and ``TRANSMISSION`` sites are active iff
    :attr:`Mesh.mask<differt.geometry.Mesh.mask>` is set for their (quad-aware)
    primitive.

    If :attr:`Mesh.mask<differt.geometry.Mesh.mask>` is :data:`None`, every
    site is considered active.

    Args:
        mesh: The scene mesh that ``sites`` was built from.
        sites: The interaction sites, see :func:`build_interaction_sites`.

    Returns:
        A boolean mask, :data:`True` for sites whose primitive is active.
    """
    from differt.em import InteractionType  # ruff: ignore[import-outside-top-level]

    if mesh.mask is None:
        return jnp.ones(sites.kind.shape[0], dtype=bool)

    is_diffraction = sites.kind == InteractionType.DIFFRACTION
    # 'sites.primitive' is a first-triangle (even) index for a quad, or a
    # plain triangle index otherwise; 'mesh.mask' is never quad-folded, so
    # index it directly, requiring both triangles of a quad to be active.
    safe_primitive = jnp.where(is_diffraction, 0, sites.primitive)
    if mesh.assume_quads:
        surface_active = mesh.mask[safe_primitive] & mesh.mask[safe_primitive + 1]
    else:
        surface_active = mesh.mask[safe_primitive]
    return jnp.where(is_diffraction, True, surface_active)
