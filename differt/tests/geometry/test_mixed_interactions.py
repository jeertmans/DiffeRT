import chex
import jax.numpy as jnp
import pytest

from differt.em import InteractionType
from differt.geometry._mesh import Mesh
from differt.geometry._paths import TracedPaths
from differt.geometry.solvers._base import _trace_path_candidates
from differt.geometry.solvers._dispatch import solve_mixed_interaction_paths


@pytest.fixture
def wedge_mesh() -> Mesh:
    # A right-angle convex wedge (like a building corner), with exactly one
    # diffraction edge, running from (1, 0, 0) to (1, 1, 0) (half-edge index 2).
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


@pytest.fixture
def wedge_and_wall_mesh(wedge_mesh: Mesh) -> Mesh:
    # The wedge above, plus a transmissive wall roughly in the x=3 plane
    # (triangle index 2).
    vertices = jnp.concatenate([
        wedge_mesh.vertices,
        jnp.array([
            [3.0, -1.0, -2.0],
            [3.0, 2.0, -2.0],
            [3.0, -1.0, 2.0],
        ]),
    ])
    triangles = jnp.concatenate([
        wedge_mesh.triangles,
        jnp.array([[4, 5, 6]]),
    ])
    return Mesh(vertices=vertices, triangles=triangles, assume_quads=False)


def _trace(
    mesh: Mesh,
    tx: jnp.ndarray,
    rx: jnp.ndarray,
    path_candidates: jnp.ndarray,
    interaction_types: jnp.ndarray,
    *,
    use_fermat: bool,
    needs_splice: bool,
) -> TracedPaths:
    return _trace_path_candidates(
        mesh,
        tx,
        rx,
        path_candidates,
        interaction_types,
        epsilon=None,
        hit_tol=None,
        min_len=None,
        smoothing_factor=None,
        confidence_threshold=0.5,
        batch_size=512,
        use_fermat=use_fermat,
        needs_splice=needs_splice,
    )


def test_pure_diffraction_lands_on_edge(wedge_mesh: Mesh) -> None:
    tx = jnp.array([[0.5, 0.5, 1.0]])
    rx = jnp.array([[2.0, 0.5, -0.5]])

    traced = _trace(
        wedge_mesh,
        tx,
        rx,
        jnp.array([[2]]),
        jnp.array([[InteractionType.DIFFRACTION]]),
        use_fermat=True,
        needs_splice=False,
    )

    assert traced.mask.item() is True
    diffraction_point = traced.vertices[0, 0, 0, 1, :]
    # On the edge line x=1, z=0.
    chex.assert_trees_all_close(
        diffraction_point[jnp.array([0, 2])], jnp.array([1.0, 0.0])
    )
    # Within the finite segment y in [0, 1].
    assert 0.0 <= diffraction_point[1] <= 1.0


def test_missing_interaction_types_defaults_to_reflection(wedge_mesh: Mesh) -> None:
    # The 'legacy' direct-call convention: omitting 'interaction_types'
    # (passing 'None') must behave exactly as if every active bounce were
    # explicitly typed as a 'REFLECTION'.
    tx = jnp.array([[0.5, 0.3, 1.0]])
    rx = jnp.array([[0.5, 0.3, 1.0]])

    traced_implicit = _trace_path_candidates(
        wedge_mesh,
        tx,
        rx,
        jnp.array([[0]]),
        None,
        epsilon=None,
        hit_tol=None,
        min_len=None,
        smoothing_factor=None,
        confidence_threshold=0.5,
        batch_size=512,
        use_fermat=False,
        needs_splice=False,
    )
    traced_explicit = _trace(
        wedge_mesh,
        tx,
        rx,
        jnp.array([[0]]),
        jnp.array([[InteractionType.REFLECTION]]),
        use_fermat=False,
        needs_splice=False,
    )

    chex.assert_trees_all_equal(traced_implicit.mask, traced_explicit.mask)
    chex.assert_trees_all_close(traced_implicit.vertices, traced_explicit.vertices)


def test_diffraction_with_smoothing_and_fermat(wedge_mesh: Mesh) -> None:
    # Soft (smoothed) edge-segment-membership check, combined with the
    # Fermat solver (used as soon as a DIFFRACTION bounce is present).
    tx = jnp.array([[0.5, 0.5, 1.0]])
    rx = jnp.array([[2.0, 0.5, -0.5]])

    traced = _trace_path_candidates(
        wedge_mesh,
        tx,
        rx,
        jnp.array([[2]]),
        jnp.array([[InteractionType.DIFFRACTION]]),
        epsilon=None,
        hit_tol=None,
        min_len=None,
        smoothing_factor=1e2,
        confidence_threshold=0.5,
        batch_size=512,
        use_fermat=True,
        needs_splice=False,
    )

    # Soft mask, close to (but not necessarily exactly) 1 for a path that
    # lands well within the finite edge segment.
    assert 0.0 < traced.mask.item() <= 1.0


def test_solve_mixed_interaction_paths_extra_fermat_kwargs(wedge_mesh: Mesh) -> None:
    # 'fermat_kwargs' (only used when 'use_fermat=True') is forwarded to,
    # and overrides the auto-derived keyword arguments passed to,
    # 'fermat_path_on_linear_objects'.
    tx = jnp.array([[0.5, 0.5, 1.0]])
    rx = jnp.array([[2.0, 0.5, -0.5]])

    default_paths = solve_mixed_interaction_paths(
        wedge_mesh,
        tx,
        rx,
        jnp.array([[2]]),
        jnp.array([[InteractionType.DIFFRACTION]]),
        use_fermat=True,
        needs_splice=False,
    )
    custom_paths = solve_mixed_interaction_paths(
        wedge_mesh,
        tx,
        rx,
        jnp.array([[2]]),
        jnp.array([[InteractionType.DIFFRACTION]]),
        use_fermat=True,
        needs_splice=False,
        fermat_kwargs={"steps": 0},
    )

    assert default_paths.shape == custom_paths.shape == (1, 1, 1, 3, 3)
    # Zero solver steps leaves the initial (unconverged) guess untouched,
    # unlike the default step count.
    assert not jnp.allclose(default_paths, custom_paths)


def test_diffraction_beyond_edge_segment_is_invalid(wedge_mesh: Mesh) -> None:
    # TX/RX far along y: the (infinite-line) Fermat optimum lands near y=5,
    # well outside the finite edge segment y in [0, 1].
    tx = jnp.array([[0.5, 5.0, 1.0]])
    rx = jnp.array([[1.5, 5.0, -1.0]])

    traced = _trace(
        wedge_mesh,
        tx,
        rx,
        jnp.array([[2]]),
        jnp.array([[InteractionType.DIFFRACTION]]),
        use_fermat=True,
        needs_splice=False,
    )

    assert traced.mask.item() is False


def test_pure_transmission_splices_straight_through(wedge_and_wall_mesh: Mesh) -> None:
    tx = jnp.array([[2.0, 0.5, -1.5]])
    rx = jnp.array([[4.0, 0.5, -1.5]])

    traced = _trace(
        wedge_and_wall_mesh,
        tx,
        rx,
        jnp.array([[2]]),
        jnp.array([[InteractionType.TRANSMISSION]]),
        use_fermat=False,
        needs_splice=True,
    )

    assert traced.mask.item() is True
    transmission_point = traced.vertices[0, 0, 0, 1, :]
    chex.assert_trees_all_close(transmission_point, jnp.array([3.0, 0.5, -1.5]))


def test_diffraction_then_transmission(wedge_and_wall_mesh: Mesh) -> None:
    tx = jnp.array([[0.5, 0.5, 1.0]])
    rx = jnp.array([[5.0, 0.5, -0.5]])

    traced = _trace(
        wedge_and_wall_mesh,
        tx,
        rx,
        jnp.array([[2, 2]]),
        jnp.array([[InteractionType.DIFFRACTION, InteractionType.TRANSMISSION]]),
        use_fermat=True,
        needs_splice=True,
    )

    assert traced.mask.item() is True
    vertices = traced.vertices[0, 0, 0]
    diffraction_point = vertices[1]
    transmission_point = vertices[2]

    chex.assert_trees_all_close(
        diffraction_point[jnp.array([0, 2])], jnp.array([1.0, 0.0]), atol=1e-6
    )
    chex.assert_trees_all_close(transmission_point[0], jnp.array(3.0), atol=1e-6)
    # The transmission point must lie on the straight segment between the
    # diffraction point and the receiver (transmission does not bend the ray).
    direction_to_rx = rx[0] - diffraction_point
    direction_to_transmission = transmission_point - diffraction_point
    cross = jnp.cross(direction_to_rx, direction_to_transmission)
    chex.assert_trees_all_close(cross, jnp.zeros(3), atol=1e-5)


def test_incidental_crossing_without_transmission_is_blocked(
    wedge_and_wall_mesh: Mesh,
) -> None:
    # Same geometry as 'test_diffraction_then_transmission', but the wall is
    # not modeled as a transmission bounce: since the diffracted ray still
    # geometrically crosses that (now purely opaque) wall on its way to the
    # receiver, the path must be correctly flagged as blocked.
    tx = jnp.array([[0.5, 0.5, 1.0]])
    rx = jnp.array([[5.0, 0.5, -0.5]])

    traced = _trace(
        wedge_and_wall_mesh,
        tx,
        rx,
        jnp.array([[2]]),
        jnp.array([[InteractionType.DIFFRACTION]]),
        use_fermat=True,
        needs_splice=False,
    )

    assert traced.mask.item() is False


def test_reflection_only_matches_legacy_defaults(wedge_mesh: Mesh) -> None:
    tx = jnp.array([[0.5, 0.5, 1.0]])
    rx = jnp.array([[1.5, 0.5, 1.0]])

    kwargs = {
        "epsilon": None,
        "hit_tol": None,
        "min_len": None,
        "smoothing_factor": None,
        "confidence_threshold": 0.5,
        "batch_size": 512,
    }
    legacy = _trace_path_candidates(
        wedge_mesh,
        tx,
        rx,
        jnp.array([[0]]),
        jnp.array([[InteractionType.REFLECTION]]),
        **kwargs,
    )
    explicit = _trace_path_candidates(
        wedge_mesh,
        tx,
        rx,
        jnp.array([[0]]),
        jnp.array([[InteractionType.REFLECTION]]),
        use_fermat=False,
        needs_splice=False,
        **kwargs,
    )
    chex.assert_trees_all_equal(legacy.vertices, explicit.vertices)
    chex.assert_trees_all_equal(legacy.mask, explicit.mask)


def test_mixed_interactions_batched_tx_rx(wedge_and_wall_mesh: Mesh) -> None:
    tx = jnp.array([[0.5, 0.5, 1.0], [0.5, 0.6, 1.0]])
    rx = jnp.array([[5.0, 0.5, -0.5], [5.0, 0.6, -0.5], [5.0, 0.7, -0.5]])

    traced = _trace(
        wedge_and_wall_mesh,
        tx,
        rx,
        jnp.array([[2, 2]]),
        jnp.array([[InteractionType.DIFFRACTION, InteractionType.TRANSMISSION]]),
        use_fermat=True,
        needs_splice=True,
    )

    assert traced.vertices.shape == (2, 3, 1, 4, 3)
    assert traced.mask.shape == (2, 3, 1)
