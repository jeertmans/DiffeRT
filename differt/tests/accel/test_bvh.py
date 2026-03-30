"""Tests for BVH acceleration structure.

Validates that BVH-accelerated intersection queries produce the same results
as the brute-force implementations, for both hard and soft (differentiable) modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest

if TYPE_CHECKING:
    from differt.scene import TriangleScene

from differt.accel import TriangleBvh
from differt.accel._accelerated import (
    bvh_first_triangles_hit_by_rays,
    bvh_rays_intersect_any_triangle,
    bvh_triangles_visible_from_vertices,
)
from differt.accel._bvh import compute_expansion_radius
from differt.rt._utils import (
    first_triangles_hit_by_rays,
    rays_intersect_any_triangle,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def single_triangle() -> jax.Array:
    return jnp.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=jnp.float32)


@pytest.fixture
def three_triangles() -> jax.Array:
    """Three triangles at different z-planes."""
    return jnp.array(
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],  # z=0
            [[0, 0, 2], [1, 0, 2], [0, 1, 2]],  # z=2
            [[5, 5, 0], [6, 5, 0], [5, 6, 0]],  # far away
        ],
        dtype=jnp.float32,
    )


@pytest.fixture
def cube_scene() -> jax.Array:
    """12-triangle unit cube."""
    faces = [
        ([0, 0, 1], [1, 0, 1], [1, 1, 1]),
        ([0, 0, 1], [1, 1, 1], [0, 1, 1]),
        ([0, 0, 0], [0, 1, 0], [1, 1, 0]),
        ([0, 0, 0], [1, 1, 0], [1, 0, 0]),
        ([0, 1, 0], [0, 1, 1], [1, 1, 1]),
        ([0, 1, 0], [1, 1, 1], [1, 1, 0]),
        ([0, 0, 0], [1, 0, 0], [1, 0, 1]),
        ([0, 0, 0], [1, 0, 1], [0, 0, 1]),
        ([1, 0, 0], [1, 1, 0], [1, 1, 1]),
        ([1, 0, 0], [1, 1, 1], [1, 0, 1]),
        ([0, 0, 0], [0, 0, 1], [0, 1, 1]),
        ([0, 0, 0], [0, 1, 1], [0, 1, 0]),
    ]
    return jnp.array(faces, dtype=jnp.float32)


@pytest.fixture
def random_scene() -> jax.Array:
    """50 random triangles in a 10x10x10 box."""
    key = jax.random.PRNGKey(42)
    return jax.random.uniform(key, (50, 3, 3), minval=0.0, maxval=10.0)


# ---------------------------------------------------------------------------
# TriangleBvh construction
# ---------------------------------------------------------------------------


class TestTriangleBvhConstruction:
    def test_single_triangle(self, single_triangle: jax.Array) -> None:
        bvh = TriangleBvh(single_triangle)
        assert bvh.num_triangles == 1
        assert bvh.num_nodes >= 1

    def test_cube(self, cube_scene: jax.Array) -> None:
        bvh = TriangleBvh(cube_scene)
        assert bvh.num_triangles == 12

    def test_random(self, random_scene: jax.Array) -> None:
        bvh = TriangleBvh(random_scene)
        assert bvh.num_triangles == 50

    def test_numpy_input(self) -> None:
        verts = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)
        bvh = TriangleBvh(verts)
        assert bvh.num_triangles == 1


# ---------------------------------------------------------------------------
# Nearest hit: BVH vs brute force
# ---------------------------------------------------------------------------


class TestNearestHit:
    def test_single_triangle_hit(self, single_triangle: jax.Array) -> None:
        bvh = TriangleBvh(single_triangle)
        origins = jnp.array([[0.1, 0.1, 1.0]])
        dirs = jnp.array([[0.0, 0.0, -1.0]])

        bvh_idx, bvh_t = bvh_first_triangles_hit_by_rays(
            origins, dirs, single_triangle, bvh=bvh
        )
        bf_idx, bf_t = first_triangles_hit_by_rays(origins, dirs, single_triangle)

        assert int(bvh_idx[0]) == int(bf_idx[0])
        np.testing.assert_allclose(float(bvh_t[0]), float(bf_t[0]), atol=1e-4)

    def test_single_triangle_miss(self, single_triangle: jax.Array) -> None:
        bvh = TriangleBvh(single_triangle)
        origins = jnp.array([[0.1, 0.1, 1.0]])
        dirs = jnp.array([[0.0, 0.0, 1.0]])  # pointing away

        bvh_idx, _bvh_t = bvh_first_triangles_hit_by_rays(
            origins, dirs, single_triangle, bvh=bvh
        )
        bf_idx, _bf_t = first_triangles_hit_by_rays(origins, dirs, single_triangle)

        assert int(bvh_idx[0]) == int(bf_idx[0]) == -1

    def test_cube_multiple_rays(self, cube_scene: jax.Array) -> None:
        bvh = TriangleBvh(cube_scene)
        origins = jnp.array(
            [
                [0.5, 0.5, 2.0],
                [0.5, 0.5, -1.0],
                [2.0, 0.5, 0.5],
                [0.5, 2.0, 0.5],
                [5.0, 5.0, 5.0],  # miss
            ],
            dtype=jnp.float32,
        )
        dirs = jnp.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],  # miss
            ],
            dtype=jnp.float32,
        )

        bvh_idx, bvh_t = bvh_first_triangles_hit_by_rays(
            origins, dirs, cube_scene, bvh=bvh
        )
        bf_idx, bf_t = first_triangles_hit_by_rays(origins, dirs, cube_scene)

        # Both should agree on hits vs misses
        bvh_hit = np.asarray(bvh_idx) >= 0
        bf_hit = np.asarray(bf_idx) >= 0
        np.testing.assert_array_equal(bvh_hit, bf_hit)

        # For hits, t values should match
        for i in range(len(origins)):
            if bvh_hit[i]:
                np.testing.assert_allclose(float(bvh_t[i]), float(bf_t[i]), atol=1e-4)

    def test_random_scene_many_rays(self, random_scene: jax.Array) -> None:
        bvh = TriangleBvh(random_scene)

        key = jax.random.PRNGKey(123)
        k1, k2 = jax.random.split(key)
        origins = jax.random.uniform(k1, (100, 3), minval=-2.0, maxval=12.0)
        dirs = jax.random.normal(k2, (100, 3))
        dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)

        bvh_idx, bvh_t = bvh_first_triangles_hit_by_rays(
            origins, dirs, random_scene, bvh=bvh
        )
        bf_idx, bf_t = first_triangles_hit_by_rays(origins, dirs, random_scene)

        bvh_hit = np.asarray(bvh_idx) >= 0
        bf_hit = np.asarray(bf_idx) >= 0
        np.testing.assert_array_equal(bvh_hit, bf_hit)

        hit_mask = bvh_hit & bf_hit
        np.testing.assert_allclose(
            np.asarray(bvh_t)[hit_mask],
            np.asarray(bf_t)[hit_mask],
            atol=1e-4,
        )

    def test_fallback_without_bvh(self, single_triangle: jax.Array) -> None:
        """Without bvh parameter, falls back to brute force."""
        origins = jnp.array([[0.1, 0.1, 1.0]])
        dirs = jnp.array([[0.0, 0.0, -1.0]])

        idx, _t = bvh_first_triangles_hit_by_rays(
            origins, dirs, single_triangle, bvh=None
        )
        assert int(idx[0]) == 0


# ---------------------------------------------------------------------------
# Any-triangle intersection: BVH vs brute force
# ---------------------------------------------------------------------------


class TestAnyIntersection:
    def test_hard_mode(self, three_triangles: jax.Array) -> None:
        bvh = TriangleBvh(three_triangles)
        # Ray from above, hits triangle at z=2
        origins = jnp.array([[0.1, 0.1, 3.0]])
        dirs = jnp.array([[0.0, 0.0, -1.0]])

        bvh_any = bvh_rays_intersect_any_triangle(
            origins, dirs, three_triangles, bvh=bvh
        )
        bf_any = rays_intersect_any_triangle(origins, dirs, three_triangles)

        assert bool(bvh_any[0]) == bool(bf_any[0])

    def test_hard_mode_miss(self, three_triangles: jax.Array) -> None:
        bvh = TriangleBvh(three_triangles)
        origins = jnp.array([[0.1, 0.1, 3.0]])
        dirs = jnp.array([[0.0, 0.0, 1.0]])  # pointing away

        bvh_any = bvh_rays_intersect_any_triangle(
            origins, dirs, three_triangles, bvh=bvh
        )
        bf_any = rays_intersect_any_triangle(origins, dirs, three_triangles)

        assert bool(bvh_any[0]) == bool(bf_any[0]) == False  # noqa: E712

    @pytest.mark.parametrize("smoothing_factor", [1.0, 10.0, 100.0])
    def test_soft_mode_matches_brute_force(
        self, three_triangles: jax.Array, smoothing_factor: float
    ) -> None:
        bvh = TriangleBvh(three_triangles)
        origins = jnp.array([[0.1, 0.1, 3.0]])
        dirs = jnp.array([[0.0, 0.0, -1.0]])

        bvh_soft = bvh_rays_intersect_any_triangle(
            origins,
            dirs,
            three_triangles,
            smoothing_factor=smoothing_factor,
            bvh=bvh,
        )
        bf_soft = rays_intersect_any_triangle(
            origins,
            dirs,
            three_triangles,
            smoothing_factor=smoothing_factor,
        )

        np.testing.assert_allclose(float(bvh_soft[0]), float(bf_soft[0]), atol=1e-3)

    def test_soft_mode_random_scene(self, random_scene: jax.Array) -> None:
        bvh = TriangleBvh(random_scene)
        key = jax.random.PRNGKey(456)
        k1, k2 = jax.random.split(key)
        origins = jax.random.uniform(k1, (20, 3), minval=0.0, maxval=10.0)
        dirs = jax.random.normal(k2, (20, 3))
        dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)

        bvh_soft = bvh_rays_intersect_any_triangle(
            origins,
            dirs,
            random_scene,
            smoothing_factor=10.0,
            bvh=bvh,
            max_candidates=256,
        )
        bf_soft = rays_intersect_any_triangle(
            origins, dirs, random_scene, smoothing_factor=10.0
        )

        np.testing.assert_allclose(np.asarray(bvh_soft), np.asarray(bf_soft), atol=1e-2)

    def test_fallback_without_bvh(self, three_triangles: jax.Array) -> None:
        # Ray from z=3 to z=-2 (length 5), triangle at z=2 is at t=0.2
        origins = jnp.array([[0.1, 0.1, 3.0]])
        dirs = jnp.array([[0.0, 0.0, -5.0]])

        result = bvh_rays_intersect_any_triangle(
            origins, dirs, three_triangles, bvh=None
        )
        assert bool(result[0])


# ---------------------------------------------------------------------------
# Expansion radius
# ---------------------------------------------------------------------------


class TestExpansionRadius:
    def test_positive(self) -> None:
        r = compute_expansion_radius(10.0, 1.0, 1e-7)
        assert r > 0

    def test_decreases_with_smoothing(self) -> None:
        r1 = compute_expansion_radius(1.0, 1.0, 1e-7)
        r2 = compute_expansion_radius(10.0, 1.0, 1e-7)
        r3 = compute_expansion_radius(100.0, 1.0, 1e-7)
        assert r1 > r2 > r3

    def test_scales_with_triangle_size(self) -> None:
        r1 = compute_expansion_radius(10.0, 1.0, 1e-7)
        r2 = compute_expansion_radius(10.0, 2.0, 1e-7)
        np.testing.assert_allclose(r2, 2 * r1)

    def test_zero_smoothing(self) -> None:
        r = compute_expansion_radius(0.0, 1.0, 1e-7)
        assert r == float("inf")


# ---------------------------------------------------------------------------
# Visibility: BVH vs brute force
# ---------------------------------------------------------------------------


class TestVisibility:
    def test_single_triangle_visible(self, single_triangle: jax.Array) -> None:
        bvh = TriangleBvh(single_triangle)
        origin = jnp.array([0.3, 0.3, 1.0])

        bvh_vis = bvh_triangles_visible_from_vertices(
            origin, single_triangle, bvh=bvh, num_rays=1000
        )
        assert bool(bvh_vis[0])  # triangle is visible from above

    def test_cube_all_visible(self, cube_scene: jax.Array) -> None:
        bvh = TriangleBvh(cube_scene)
        origin = jnp.array([0.5, 0.5, 2.0])  # above the cube

        bvh_vis = bvh_triangles_visible_from_vertices(
            origin, cube_scene, bvh=bvh, num_rays=10000
        )
        # From above, the top face triangles should be visible
        assert int(bvh_vis.sum()) >= 2  # at least the top face

    def test_matches_brute_force(self, cube_scene: jax.Array) -> None:
        from differt.rt import triangles_visible_from_vertices  # noqa: PLC0415

        bvh = TriangleBvh(cube_scene)
        origin = jnp.array([0.5, 0.5, 2.0])

        bvh_vis = bvh_triangles_visible_from_vertices(
            origin, cube_scene, bvh=bvh, num_rays=10000
        )
        bf_vis = triangles_visible_from_vertices(origin, cube_scene, num_rays=10000)

        # Both should see approximately the same set (statistical)
        bvh_count = int(bvh_vis.sum())
        bf_count = int(bf_vis.sum())
        assert abs(bvh_count - bf_count) <= 2  # allow small difference

    def test_fallback_without_bvh(self, single_triangle: jax.Array) -> None:
        origin = jnp.array([0.3, 0.3, 1.0])
        vis = bvh_triangles_visible_from_vertices(
            origin, single_triangle, bvh=None, num_rays=1000
        )
        assert bool(vis[0])

    def test_multiple_origins(self, cube_scene: jax.Array) -> None:
        bvh = TriangleBvh(cube_scene)
        origins = jnp.array([
            [0.5, 0.5, 2.0],  # above
            [0.5, 0.5, -1.0],  # below
        ])

        bvh_vis = bvh_triangles_visible_from_vertices(
            origins, cube_scene, bvh=bvh, num_rays=10000
        )
        assert bvh_vis.shape == (2, 12)
        assert int(bvh_vis[0].sum()) >= 2  # top visible
        assert int(bvh_vis[1].sum()) >= 2  # bottom visible


# ---------------------------------------------------------------------------
# compute_paths integration
# ---------------------------------------------------------------------------


def _has_xla_ffi() -> bool:
    """Check if differt-core was built with xla-ffi feature."""
    try:
        from differt.accel._ffi import _ensure_registered  # noqa: PLC0415

        _ensure_registered()
    except (ImportError, AttributeError):
        return False
    return True


_requires_ffi = pytest.mark.skipif(
    not _has_xla_ffi(),
    reason="differt-core not built with xla-ffi feature",
)


@_requires_ffi
class TestComputePathsBvh:
    def test_hybrid_with_bvh(self) -> None:
        import equinox as eqx  # noqa: PLC0415

        from differt.scene import TriangleScene  # noqa: PLC0415

        scene = TriangleScene.load_xml(
            "differt/src/differt/scene/scenes/simple_reflector/simple_reflector.xml"
        )
        scene = eqx.tree_at(
            lambda s: s.transmitters, scene, jnp.array([[0.5, 0.5, 1.0]])
        )
        scene = eqx.tree_at(lambda s: s.receivers, scene, jnp.array([[-0.5, 0.5, 0.5]]))
        bvh = scene.build_bvh()

        paths_bvh = scene.compute_paths(order=1, method="hybrid", bvh=bvh)
        paths_bf = scene.compute_paths(order=1, method="hybrid")

        # Both should find the same valid paths
        assert paths_bvh.mask.shape == paths_bf.mask.shape
        np.testing.assert_array_equal(
            np.asarray(paths_bvh.mask), np.asarray(paths_bf.mask)
        )

    def test_exhaustive_with_bvh_ffi(self) -> None:
        """Exhaustive method uses BVH FFI for blocking check."""
        import equinox as eqx  # noqa: PLC0415

        from differt.scene import TriangleScene  # noqa: PLC0415

        scene = TriangleScene.load_xml(
            "differt/src/differt/scene/scenes/simple_reflector/simple_reflector.xml"
        )
        scene = eqx.tree_at(
            lambda s: s.transmitters, scene, jnp.array([[0.5, 0.5, 1.0]])
        )
        scene = eqx.tree_at(lambda s: s.receivers, scene, jnp.array([[-0.5, 0.5, 0.5]]))
        bvh = scene.build_bvh()

        # BVH should give same results as brute force for hard mode
        paths_bvh = scene.compute_paths(order=1, method="exhaustive", bvh=bvh)
        paths_bf = scene.compute_paths(order=1, method="exhaustive")

        np.testing.assert_array_equal(
            np.asarray(paths_bvh.mask), np.asarray(paths_bf.mask)
        )

    def test_sbr_with_bvh_ffi(self) -> None:
        """SBR method uses BVH FFI in the bounce loop."""
        import equinox as eqx  # noqa: PLC0415

        from differt.scene import TriangleScene  # noqa: PLC0415

        scene = TriangleScene.load_xml("differt/src/differt/scene/scenes/box/box.xml")
        scene = eqx.tree_at(
            lambda s: s.transmitters, scene, jnp.array([[0.5, 0.5, 2.0]])
        )
        scene = eqx.tree_at(lambda s: s.receivers, scene, jnp.array([[0.5, 0.5, -1.0]]))
        bvh = scene.build_bvh()

        paths_bvh = scene.compute_paths(order=1, method="sbr", bvh=bvh, num_rays=1000)
        paths_bf = scene.compute_paths(order=1, method="sbr", num_rays=1000)
        # Both should produce SBRPaths with same shape and mostly matching data.
        # Small differences are expected due to different Moller-Trumbore epsilon
        # (BVH uses 1e-8, brute-force uses ~1.2e-6 for f32).
        assert type(paths_bvh).__name__ == "SBRPaths"
        assert type(paths_bf).__name__ == "SBRPaths"
        assert paths_bvh.vertices.shape == paths_bf.vertices.shape
        assert paths_bvh.objects.shape == paths_bf.objects.shape
        # Most object indices should agree (allow small fraction to differ)
        objs_bvh = np.asarray(paths_bvh.objects)
        objs_bf = np.asarray(paths_bf.objects)
        match_frac = np.mean(objs_bvh == objs_bf)
        assert match_frac > 0.95, f"Object indices match only {match_frac:.1%}"

    def test_exhaustive_matches_without_bvh(self) -> None:
        """Exhaustive with BVH produces same results as without."""
        import equinox as eqx  # noqa: PLC0415

        from differt.scene import TriangleScene  # noqa: PLC0415

        scene = TriangleScene.load_xml(
            "differt/src/differt/scene/scenes/simple_reflector/simple_reflector.xml"
        )
        scene = eqx.tree_at(
            lambda s: s.transmitters, scene, jnp.array([[0.5, 0.5, 1.0]])
        )
        scene = eqx.tree_at(lambda s: s.receivers, scene, jnp.array([[-0.5, 0.5, 0.5]]))
        bvh = scene.build_bvh()

        paths_bvh = scene.compute_paths(order=1, method="exhaustive", bvh=bvh)
        paths_bf = scene.compute_paths(order=1, method="exhaustive")

        np.testing.assert_array_equal(
            np.asarray(paths_bvh.mask), np.asarray(paths_bf.mask)
        )


# ---------------------------------------------------------------------------
# Coverage: batched ray inputs (_bvh.py ndim > 2 branches)
# ---------------------------------------------------------------------------


class TestBatchedRays:
    def test_nearest_hit_3d_origins(self, single_triangle: jax.Array) -> None:
        """nearest_hit with ndim > 2 triggers the reshape branch."""
        bvh = TriangleBvh(single_triangle)
        # Shape (2, 1, 3) -- batch dimension
        origins = jnp.array([[[0.1, 0.1, 1.0]], [[0.1, 0.1, 1.0]]], dtype=jnp.float32)
        dirs = jnp.array([[[0.0, 0.0, -1.0]], [[0.0, 0.0, 1.0]]], dtype=jnp.float32)
        idx, _t = bvh.nearest_hit(origins, dirs)
        assert idx.shape == (2, 1)
        assert int(idx[0, 0]) == 0  # hit
        assert int(idx[1, 0]) == -1  # miss (pointing away)

    def test_get_candidates_3d_origins(self, single_triangle: jax.Array) -> None:
        """get_candidates with ndim > 2 triggers the reshape branch."""
        bvh = TriangleBvh(single_triangle)
        origins = jnp.array([[[0.1, 0.1, 1.0]], [[5.0, 5.0, 1.0]]], dtype=jnp.float32)
        dirs = jnp.array([[[0.0, 0.0, -1.0]], [[0.0, 0.0, -1.0]]], dtype=jnp.float32)
        max_cands = 8
        idx, counts = bvh.get_candidates(
            origins, dirs, expansion=0.0, max_candidates=max_cands
        )
        assert idx.shape == (2, 1, max_cands)
        assert counts.shape == (2, 1)
        assert int(counts[0, 0]) >= 1  # near triangle
        assert int(counts[1, 0]) == 0  # far away, no candidates


# ---------------------------------------------------------------------------
# Coverage: expansion radius edge case
# ---------------------------------------------------------------------------


class TestExpansionRadiusEdgeCases:
    def test_negative_smoothing(self) -> None:
        r = compute_expansion_radius(-5.0, 1.0, 1e-7)
        assert r == float("inf")


# ---------------------------------------------------------------------------
# Coverage: _accelerated.py uncovered branches
# ---------------------------------------------------------------------------


class TestAcceleratedBranches:
    def test_soft_mode_large_expansion_fallback(
        self, three_triangles: jax.Array
    ) -> None:
        """Very small smoothing_factor -> huge expansion -> brute-force fallback."""
        bvh = TriangleBvh(three_triangles)
        origins = jnp.array([[0.1, 0.1, 3.0]])
        dirs = jnp.array([[0.0, 0.0, -1.0]])

        # smoothing_factor=0.001 produces expansion >> scene diagonal
        result = bvh_rays_intersect_any_triangle(
            origins, dirs, three_triangles, smoothing_factor=0.001, bvh=bvh
        )
        bf_result = rays_intersect_any_triangle(
            origins, dirs, three_triangles, smoothing_factor=0.001
        )
        np.testing.assert_allclose(float(result[0]), float(bf_result[0]), atol=1e-3)

    def test_soft_mode_max_candidates_exceeded(self, random_scene: jax.Array) -> None:
        """max_candidates=1 with many overlapping triangles -> warning + fallback."""
        bvh = TriangleBvh(random_scene)
        key = jax.random.PRNGKey(789)
        k1, k2 = jax.random.split(key)
        origins = jax.random.uniform(k1, (10, 3), minval=0.0, maxval=10.0)
        dirs = jax.random.normal(k2, (10, 3))
        dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)

        with pytest.warns(UserWarning, match="BVH candidate count"):
            result = bvh_rays_intersect_any_triangle(
                origins,
                dirs,
                random_scene,
                smoothing_factor=100.0,
                bvh=bvh,
                max_candidates=1,
            )
        assert result.shape == (10,)

    def test_hard_mode_with_active_triangles(self, three_triangles: jax.Array) -> None:
        """active_triangles mask in hard mode for bvh_rays_intersect_any_triangle."""
        bvh = TriangleBvh(three_triangles)
        # Ray from z=3 pointing down with length 5 (t < 1 for triangles at z=2 and z=0)
        origins = jnp.array([[0.1, 0.1, 3.0]])
        dirs = jnp.array([[0.0, 0.0, -5.0]])

        # All active: should hit
        active_all = jnp.array([True, True, True])
        result_all = bvh_rays_intersect_any_triangle(
            origins, dirs, three_triangles, active_triangles=active_all, bvh=bvh
        )
        assert bool(result_all[0])

        # Only the far-away triangle active: should miss
        active_far = jnp.array([False, False, True])
        result_far = bvh_rays_intersect_any_triangle(
            origins, dirs, three_triangles, active_triangles=active_far, bvh=bvh
        )
        assert not bool(result_far[0])

    def test_soft_mode_with_active_triangles(self, three_triangles: jax.Array) -> None:
        """active_triangles mask in soft mode for bvh_rays_intersect_any_triangle."""
        bvh = TriangleBvh(three_triangles)
        origins = jnp.array([[0.1, 0.1, 3.0]])
        dirs = jnp.array([[0.0, 0.0, -1.0]])

        active = jnp.array([True, True, False])
        result = bvh_rays_intersect_any_triangle(
            origins,
            dirs,
            three_triangles,
            active_triangles=active,
            smoothing_factor=100.0,
            bvh=bvh,
        )
        assert result.shape == (1,)
        assert float(result[0]) > 0  # should detect the hit

    def test_first_hit_with_active_triangles(self, three_triangles: jax.Array) -> None:
        """active_triangles mask for bvh_first_triangles_hit_by_rays."""
        bvh = TriangleBvh(three_triangles)
        # Ray from z=3 pointing down: nearest active hit changes with mask
        origins = jnp.array([[0.1, 0.1, 3.0]])
        dirs = jnp.array([[0.0, 0.0, -5.0]])  # length 5 so t < 1 for all hits

        # Only triangle 0 (z=0) active, triangle 1 (z=2) inactive
        active = jnp.array([True, False, True])
        idx, t = bvh_first_triangles_hit_by_rays(
            origins, dirs, three_triangles, active_triangles=active, bvh=bvh
        )
        assert int(idx[0]) == 0  # nearest active is z=0
        np.testing.assert_allclose(float(t[0]), 0.6, atol=1e-4)  # 3.0/5.0

    def test_visibility_with_active_triangles(self, single_triangle: jax.Array) -> None:
        """active_triangles mask for bvh_triangles_visible_from_vertices."""
        bvh = TriangleBvh(single_triangle)
        origin = jnp.array([0.3, 0.3, 1.0])

        active = jnp.array([True])
        vis_active = bvh_triangles_visible_from_vertices(
            origin, single_triangle, active_triangles=active, bvh=bvh, num_rays=1000
        )
        assert bool(vis_active[0])

        inactive = jnp.array([False])
        vis_inactive = bvh_triangles_visible_from_vertices(
            origin, single_triangle, active_triangles=inactive, bvh=bvh, num_rays=1000
        )
        assert not bool(vis_inactive[0])


# ---------------------------------------------------------------------------
# Coverage: _ffi.py ensure_registered branches
# ---------------------------------------------------------------------------


class TestFfiRegistration:
    def test_ensure_registered_idempotent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Second call to _ensure_registered short-circuits."""
        import differt.accel._ffi as ffi_mod  # noqa: PLC0415

        monkeypatch.setattr(ffi_mod, "_FFI_REGISTERED", True)
        # Should return immediately without touching differt_core
        ffi_mod._ensure_registered()  # noqa: SLF001

    def test_ensure_registered_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing xla-ffi feature raises ImportError with helpful message."""
        import sys  # noqa: PLC0415

        import differt.accel._ffi as ffi_mod  # noqa: PLC0415

        monkeypatch.setattr(ffi_mod, "_FFI_REGISTERED", False)

        # Remove differt_core._differt_core from sys.modules so the import fails
        monkeypatch.delitem(sys.modules, "differt_core._differt_core", raising=False)
        monkeypatch.setitem(sys.modules, "differt_core._differt_core", None)

        with pytest.raises(ImportError, match="BVH XLA FFI not available"):
            ffi_mod._ensure_registered()  # noqa: SLF001


# ---------------------------------------------------------------------------
# Coverage: _triangle_scene.py build_bvh
# ---------------------------------------------------------------------------


class TestBuildBvh:
    def test_build_bvh_from_scene(self) -> None:
        from differt.scene import TriangleScene  # noqa: PLC0415

        scene = TriangleScene.load_xml(
            "differt/src/differt/scene/scenes/simple_reflector/simple_reflector.xml"
        )
        bvh = scene.build_bvh()
        assert bvh.num_triangles == scene.mesh.num_triangles
        assert bvh.num_nodes >= 1


# ---------------------------------------------------------------------------
# Coverage: soft-mode BVH in _compute_paths
# ---------------------------------------------------------------------------


@_requires_ffi
class TestSoftModeBvhComputePaths:
    """Test differentiable BVH acceleration in compute_paths."""

    @staticmethod
    def _make_scene() -> TriangleScene:
        import equinox as eqx  # noqa: PLC0415

        from differt.scene import TriangleScene  # noqa: PLC0415

        scene = TriangleScene.load_xml(
            "differt/src/differt/scene/scenes/simple_reflector/simple_reflector.xml"
        )
        scene = eqx.tree_at(
            lambda s: s.transmitters, scene, jnp.array([[0.5, 0.5, 1.0]])
        )
        return eqx.tree_at(lambda s: s.receivers, scene, jnp.array([[-0.5, 0.5, 0.5]]))

    def test_soft_bvh_matches_brute_force(self) -> None:
        """Soft mode with BVH should match brute force within tolerance."""
        scene = self._make_scene()
        bvh = scene.build_bvh()

        for sf in [1.0, 10.0, 100.0]:
            paths_bvh = scene.compute_paths(order=1, smoothing_factor=sf, bvh=bvh)
            paths_bf = scene.compute_paths(order=1, smoothing_factor=sf)

            np.testing.assert_allclose(
                np.asarray(paths_bvh.mask),
                np.asarray(paths_bf.mask),
                atol=1e-3,
                err_msg=f"Mismatch at smoothing_factor={sf}",
            )

    def test_soft_bvh_gradient_flow(self) -> None:
        """Gradients should flow through the soft BVH path."""
        import equinox as eqx  # noqa: PLC0415

        from differt.scene import TriangleScene  # noqa: PLC0415

        scene = TriangleScene.load_xml(
            "differt/src/differt/scene/scenes/simple_reflector/simple_reflector.xml"
        )
        tx = jnp.array([[0.5, 0.5, 1.0]])
        scene = eqx.tree_at(lambda s: s.transmitters, scene, tx)
        scene = eqx.tree_at(lambda s: s.receivers, scene, jnp.array([[-0.5, 0.5, 0.5]]))
        bvh = scene.build_bvh()

        def loss_fn(tx_pos: jax.Array) -> jax.Array:
            s = eqx.tree_at(lambda s: s.transmitters, scene, tx_pos)
            paths = s.compute_paths(order=1, smoothing_factor=10.0, bvh=bvh)
            return jnp.sum(paths.mask)

        grad = jax.grad(loss_fn)(tx)
        assert jnp.all(jnp.isfinite(grad)), f"Non-finite gradients: {grad}"

    def test_soft_bvh_expansion_fallback(self) -> None:
        """Very small smoothing_factor should fall back to brute force."""
        scene = self._make_scene()
        bvh = scene.build_bvh()

        # smoothing_factor=0.001 produces huge expansion > scene diagonal
        paths_bvh = scene.compute_paths(order=1, smoothing_factor=0.001, bvh=bvh)
        paths_bf = scene.compute_paths(order=1, smoothing_factor=0.001)

        np.testing.assert_allclose(
            np.asarray(paths_bvh.mask),
            np.asarray(paths_bf.mask),
            atol=1e-6,
        )
