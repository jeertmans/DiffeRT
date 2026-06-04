import chex
import jax.numpy as jnp
import pytest

from differt.geometry._triangle_mesh import TriangleMesh


def test_boundary_edges() -> None:
    # A single triangle has 3 boundary edges, 0 diffraction edges
    vertices = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    triangles = jnp.array([[0, 1, 2]])
    mesh = TriangleMesh(vertices=vertices, triangles=triangles)

    assert mesh.diffraction_edges_mask.shape == (1, 3)
    assert not jnp.any(mesh.diffraction_edges_mask)
    assert mesh.diffraction_edges.shape[0] == 0


def test_coplanar_edges() -> None:
    # Two coplanar triangles forming a quad
    vertices = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    # Triangle 1: 0, 1, 2. Triangle 2: 1, 3, 2.
    # Shared edge between 1 and 2 is [1, 2] which is sorted (1, 2)
    triangles = jnp.array([
        [0, 1, 2],
        [1, 3, 2],
    ])
    mesh = TriangleMesh(vertices=vertices, triangles=triangles, assume_quads=False)

    # They share an edge but they are coplanar, so they should NOT be diffraction edges.
    assert not jnp.any(mesh.diffraction_edges_mask)


def test_convex_and_concave_wedges() -> None:
    # A right-angle convex wedge (like a building corner)
    # Triangle 1 on the top face (z = 0, normal [0, 0, 1])
    # Triangle 2 on the side face (x = 1, normal [1, 0, 0])
    vertices = jnp.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [1.0, 0.0, -1.0],  # 3
    ])
    # Triangle 1: (0, 1, 2) -> normal [0, 0, 1]
    # Triangle 2: (1, 3, 2) -> normal [1, 0, 0]
    triangles = jnp.array([
        [0, 1, 2],
        [1, 3, 2],
    ])
    mesh = TriangleMesh(vertices=vertices, triangles=triangles, assume_quads=False)

    # Edge (1, 2) is shared.
    # Normal 1 = [0, 0, 1], Normal 2 = [1, 0, 0].
    # Cos angle is 0, so angle phi is pi/2.
    # The corner is convex, so exterior angle is 1.5 * pi. n = 1.5.
    mask = mesh.diffraction_edges_mask
    assert (
        jnp.sum(mask) == 2
    )  # shared edge is represented once for each of the two adjacent triangles

    n = mesh.wedge_parameters
    assert n.shape == (2,)
    chex.assert_trees_all_close(n, jnp.array([1.5, 1.5]))


def test_assume_quads() -> None:
    # Two coplanar triangles forming a quad
    vertices = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    triangles = jnp.array([
        [0, 1, 2],
        [1, 3, 2],
    ])
    mesh = TriangleMesh(vertices=vertices, triangles=triangles, assume_quads=True)

    # The shared diagonal should be ignored because assume_quads=True.
    adj_t, _ = mesh._connectivity  # noqa: SLF001
    assert jnp.all(adj_t == -1)


def test_non_manifold_edges() -> None:
    # Three triangles sharing a single edge
    vertices = jnp.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [0.0, 0.0, 1.0],  # 3
        [0.0, -1.0, 0.0],  # 4
    ])
    triangles = jnp.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 1, 4],
    ])

    mesh = TriangleMesh(vertices=vertices, triangles=triangles)
    with pytest.warns(UserWarning, match="The mesh contains non-manifold edges"):
        _ = mesh.diffraction_edges_mask
