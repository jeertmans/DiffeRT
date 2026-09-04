import chex
import equinox as eqx
import jax.numpy as jnp
import pytest

from differt.em import InteractionType
from differt.geometry._interaction_sites import (
    build_interaction_sites,
    interaction_sites_mesh_mask,
    interaction_sites_valid_mask,
)
from differt.geometry._mesh import Mesh


@pytest.fixture
def wedge_mesh() -> Mesh:
    # A right-angle convex wedge (like a building corner), with exactly one
    # diffraction edge (the shared edge between the two triangles).
    vertices = jnp.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [1.0, 0.0, -1.0],  # 3
    ])
    triangles = jnp.array([
        [0, 1, 2],
        [1, 3, 2],
    ])
    return Mesh(vertices=vertices, triangles=triangles, assume_quads=False)


def test_reflection_only_matches_num_primitives(wedge_mesh: Mesh) -> None:
    sites = build_interaction_sites(wedge_mesh, frozenset({InteractionType.REFLECTION}))
    chex.assert_trees_all_equal(
        sites.kind,
        jnp.full((wedge_mesh.num_primitives,), InteractionType.REFLECTION),
    )
    chex.assert_trees_all_equal(sites.primitive, jnp.arange(wedge_mesh.num_primitives))


def test_combines_surface_interaction_types(wedge_mesh: Mesh) -> None:
    allowed = frozenset({
        InteractionType.REFLECTION,
        InteractionType.SCATTERING,
        InteractionType.TRANSMISSION,
    })
    sites = build_interaction_sites(wedge_mesh, allowed)

    num_primitives = wedge_mesh.num_primitives
    assert sites.kind.shape == (3 * num_primitives,)
    for i, interaction_type in enumerate((
        InteractionType.REFLECTION,
        InteractionType.SCATTERING,
        InteractionType.TRANSMISSION,
    )):
        chunk = sites.kind[i * num_primitives : (i + 1) * num_primitives]
        chex.assert_trees_all_equal(
            chunk, jnp.full((num_primitives,), interaction_type)
        )


def test_diffraction_sites_span_half_edges(wedge_mesh: Mesh) -> None:
    sites = build_interaction_sites(
        wedge_mesh, frozenset({InteractionType.DIFFRACTION})
    )
    num_half_edges = wedge_mesh.num_triangles * 3
    assert sites.kind.shape == (num_half_edges,)
    chex.assert_trees_all_equal(
        sites.kind, jnp.full((num_half_edges,), InteractionType.DIFFRACTION)
    )
    chex.assert_trees_all_equal(sites.primitive, jnp.arange(num_half_edges))


def test_diffraction_valid_mask_selects_exactly_one_edge(wedge_mesh: Mesh) -> None:
    sites = build_interaction_sites(
        wedge_mesh, frozenset({InteractionType.DIFFRACTION})
    )
    valid = interaction_sites_valid_mask(wedge_mesh, sites)

    assert jnp.sum(valid) == 2  # the shared edge, seen from both half-edges

    expected = wedge_mesh.diffraction_edges_mask.ravel()
    chex.assert_trees_all_equal(valid, expected)


def test_reflection_sites_are_always_valid(wedge_mesh: Mesh) -> None:
    sites = build_interaction_sites(wedge_mesh, frozenset({InteractionType.REFLECTION}))
    valid = interaction_sites_valid_mask(wedge_mesh, sites)
    assert jnp.all(valid)


def test_mixed_allowed_interactions_valid_mask(wedge_mesh: Mesh) -> None:
    allowed = frozenset({InteractionType.REFLECTION, InteractionType.DIFFRACTION})
    sites = build_interaction_sites(wedge_mesh, allowed)
    valid = interaction_sites_valid_mask(wedge_mesh, sites)

    num_primitives = wedge_mesh.num_primitives
    assert jnp.all(valid[:num_primitives])  # reflection sites
    chex.assert_trees_all_equal(
        valid[num_primitives:], wedge_mesh.diffraction_edges_mask.ravel()
    )


def test_empty_allowed_interactions_raises(wedge_mesh: Mesh) -> None:
    with pytest.raises(ValueError, match="at least one InteractionType"):
        build_interaction_sites(wedge_mesh, frozenset())


def test_unsupported_interaction_type_raises(wedge_mesh: Mesh) -> None:
    with pytest.raises(ValueError, match="Unsupported interaction type"):
        build_interaction_sites(wedge_mesh, frozenset({InteractionType.RIS}))


def test_reflection_primitives_are_first_triangle_index_when_assume_quads() -> None:
    # 2 quads (4 triangles): primitives should be the *even* (first-triangle)
    # index of each quad, i.e., [0, 2], not the raw quad index [0, 1].
    vertices = jnp.zeros((6, 3))
    triangles = jnp.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
    mesh = Mesh(vertices=vertices, triangles=triangles, assume_quads=True)

    sites = build_interaction_sites(mesh, frozenset({InteractionType.REFLECTION}))
    chex.assert_trees_all_equal(sites.primitive, jnp.array([0, 2]))


def test_mesh_mask_requires_both_triangles_of_a_quad_active() -> None:
    vertices = jnp.zeros((6, 3))
    triangles = jnp.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
    # First quad: both triangles active. Second quad: only one active.
    mask = jnp.array([True, True, True, False])
    mesh = Mesh(vertices=vertices, triangles=triangles, assume_quads=True, mask=mask)

    sites = build_interaction_sites(mesh, frozenset({InteractionType.REFLECTION}))
    active = interaction_sites_mesh_mask(mesh, sites)
    chex.assert_trees_all_equal(active, jnp.array([True, False]))


def test_mesh_mask_none_means_all_sites_active(wedge_mesh: Mesh) -> None:
    sites = build_interaction_sites(wedge_mesh, frozenset({InteractionType.REFLECTION}))
    active = interaction_sites_mesh_mask(wedge_mesh, sites)
    assert jnp.all(active)


def test_mesh_mask_diffraction_sites_always_active(wedge_mesh: Mesh) -> None:
    mask = jnp.array([False, False])  # every triangle inactive
    mesh = eqx.tree_at(lambda m: m.mask, wedge_mesh, mask, is_leaf=lambda x: x is None)
    sites = build_interaction_sites(mesh, frozenset({InteractionType.DIFFRACTION}))
    active = interaction_sites_mesh_mask(mesh, sites)
    assert jnp.all(active)
