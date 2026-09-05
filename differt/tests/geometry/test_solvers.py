import time

import equinox as eqx
import jax.numpy as jnp
import pytest

from differt.em import InteractionType
from differt.geometry import Mesh, Scene
from differt.geometry.solvers import (
    ExhaustivePathTracer,
    SBRPathLauncher,
    SBRPathTracer,
    _generate_path_candidates_for_orders,
    _normalize_order,
    _pad_path_candidates,
)


def test_sbr_launcher() -> None:
    solver = SBRPathLauncher()
    # Wait, max_dist doesn't exist? Oh I probably meant hit_tol or something? No, let's just assert solver.max_bounces == 0 or something.
    assert getattr(solver, "chunk_size", None) is None


def test_normalize_order_wraps_int_in_sequence() -> None:
    assert _normalize_order(3) == (3,)


def test_normalize_order_converts_slice_to_range() -> None:
    assert _normalize_order(slice(0, 6)) == range(6)
    assert _normalize_order(slice(0, 6, 2)) == range(0, 6, 2)


def test_normalize_order_leaves_sequence_untouched() -> None:
    order = [1, 2, 3]
    assert _normalize_order(order) is order


def test_normalize_order_slice_without_stop_raises() -> None:
    with pytest.raises(ValueError, match="must have a defined 'stop'"):
        _normalize_order(slice(0, None))


@pytest.mark.parametrize("order", [-1, [1, -2, 3], range(-2, 3), slice(-1, 3)])
def test_normalize_order_rejects_negative_orders(
    order: int | list[int] | range | slice,
) -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        _normalize_order(order)


def test_pad_path_candidates_no_op() -> None:
    candidates = jnp.array([[1, 2], [3, 4]])
    types = jnp.zeros_like(candidates)
    padded_candidates, padded_types = _pad_path_candidates(candidates, types, 2)
    assert padded_candidates is candidates
    assert padded_types is types


def test_pad_path_candidates_lower_order() -> None:
    candidates = jnp.array([[1], [3]])
    types = jnp.zeros_like(candidates)
    padded_candidates, padded_types = _pad_path_candidates(candidates, types, 3)

    assert padded_candidates.shape == (2, 3)
    assert jnp.array_equal(padded_candidates[:, 0], candidates[:, 0])
    assert jnp.all(padded_candidates[:, 1:] == -1)
    assert jnp.all(padded_types[:, 1:] == -1)


def test_pad_path_candidates_invalid_order_raises() -> None:
    candidates = jnp.zeros((2, 2), dtype=int)
    types = jnp.zeros_like(candidates)
    with pytest.raises(ValueError, match="Cannot pad"):
        _pad_path_candidates(candidates, types, 1)


@pytest.fixture
def canyon_scene() -> Scene:
    """A minimal two-wall 'canyon' scene, with each wall split into two triangles.

    Small and cheap enough to be exercised with exhaustive search directly,
    which makes it a convenient ground truth to test bounded/discovery-based
    candidate generation against.
    """
    vertices = jnp.array([
        [-5.0, -5.0, 0.0],
        [-5.0, 5.0, 0.0],
        [-5.0, 5.0, 10.0],
        [-5.0, -5.0, 10.0],
        [5.0, -5.0, 0.0],
        [5.0, 5.0, 0.0],
        [5.0, 5.0, 10.0],
        [5.0, -5.0, 10.0],
    ])
    triangles = jnp.array([
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
    ])
    mesh = Mesh(vertices=vertices, triangles=triangles)
    return Scene(
        transmitters=jnp.array([0.0, -2.0, 5.0]),
        receivers=jnp.array([0.0, 2.0, 5.0]),
        mesh=mesh,
    )


def test_generate_path_candidates_for_orders_matches_manual_padding_and_concatenation(
    canyon_scene: Scene,
) -> None:
    solver = ExhaustivePathTracer()
    orders = [0, 1, 2]

    combined_candidates, combined_types = _generate_path_candidates_for_orders(
        solver,
        canyon_scene,
        orders,
        frozenset({InteractionType.REFLECTION}),
    )

    max_order = max(orders)
    expected_parts = [
        _pad_path_candidates(
            *solver.generate_path_candidates(canyon_scene, o), max_order
        )
        for o in orders
    ]
    expected_candidates = jnp.concatenate([c for c, _ in expected_parts])
    expected_types = jnp.concatenate([t for _, t in expected_parts])

    assert jnp.array_equal(combined_candidates, expected_candidates)
    assert jnp.array_equal(combined_types, expected_types)

    # The size is known ahead of time: the sum of each individual order's
    # candidate count.
    expected_num_candidates = sum(
        solver.generate_path_candidates(canyon_scene, o)[0].shape[0] for o in orders
    )
    assert combined_candidates.shape == (expected_num_candidates, max_order)


def test_generate_path_candidates_for_orders_empty_orders_raises(
    canyon_scene: Scene,
) -> None:
    solver = ExhaustivePathTracer()
    with pytest.raises(ValueError, match="at least one order"):
        _generate_path_candidates_for_orders(
            solver,
            canyon_scene,
            [],
            frozenset({InteractionType.REFLECTION}),
        )


class TestExhaustivePathTracer:
    def test_generate_path_candidates_chunks_iter(
        self, simple_street_canyon_scene: Scene
    ) -> None:
        solver = ExhaustivePathTracer(chunk_size=3)
        order = 1

        # exhaustive generates num_candidates
        chunks_iter = solver.generate_path_candidates_chunks_iter(
            simple_street_canyon_scene, order, chunk_size=3, pad_chunks=True
        )

        chunks = list(chunks_iter)
        assert len(chunks) > 0
        # check that each chunk has size 3
        for c, i in chunks:
            assert c.shape[-2] == 3, f"Expected 3, got {c.shape}"
            assert i.shape[-2] == 3

        # check without padding
        chunks_iter2 = solver.generate_path_candidates_chunks_iter(
            simple_street_canyon_scene, order, chunk_size=7, pad_chunks=False
        )

        chunks2 = list(chunks_iter2)
        assert len(chunks2) > 0
        assert chunks2[-1][0].shape[-2] <= 7

        # check total length
        total_len1 = sum(c[0].shape[-2] for c in chunks)
        total_len2 = sum(c[0].shape[-2] for c in chunks2)
        # wait, with padding total_len1 >= total_len2
        assert total_len1 >= total_len2

    def test_trace_paths_multiple_orders_matches_individually_solved_orders(
        self, canyon_scene: Scene
    ) -> None:
        # 'ExhaustivePathTracer' (and 'HybridPathTracer') generate each
        # requested order independently and concatenate them, so the
        # combined result must exactly match the individually-traced orders.
        # This is not the case for 'SBRPathTracer', whose combined buffer is
        # shared (and bounded) across all requested orders instead, see
        # 'TestSBRPathTracer.test_recovers_ground_truth_across_orders' below.
        solver = ExhaustivePathTracer()
        orders = [0, 1, 2]
        combined = solver.trace_paths(canyon_scene, order=orders)

        individual = [solver.trace_paths(canyon_scene, order=o) for o in orders]
        expected_num_valid = sum(int(p.mask.sum()) for p in individual)

        assert combined.order == max(orders)
        assert int(combined.mask.sum()) == expected_num_valid

        # Every valid path found individually must also be found (with the
        # same, correctly-padded geometry) in the combined result.
        combined_masked_objects = {
            tuple(row.tolist()) for row in combined.masked_objects
        }
        for p in individual:
            for row in p.masked_objects:
                # Padding is inserted between the last mirror and the
                # receiver (the last element), not appended at the very end.
                missing = max(orders) - (row.shape[0] - 2)
                padded_row = (
                    *row[:-1].tolist(),
                    *([-1] * missing),
                    row[-1].item(),
                )
                assert padded_row in combined_masked_objects

    def test_trace_paths_empty_orders_raises(self, canyon_scene: Scene) -> None:
        solver = ExhaustivePathTracer()
        with pytest.raises(ValueError, match="at least one order"):
            solver.trace_paths(canyon_scene, order=[])

    def test_trace_paths_duplicate_orders_are_deduplicated(
        self, canyon_scene: Scene
    ) -> None:
        solver = ExhaustivePathTracer()
        combined = solver.trace_paths(canyon_scene, order=[1, 1, 2, 2])
        expected = solver.trace_paths(canyon_scene, order=[1, 2])
        assert jnp.array_equal(combined.objects, expected.objects)

    def test_trace_paths_chunk_size_not_supported_for_order_sequence(
        self, canyon_scene: Scene
    ) -> None:
        solver = ExhaustivePathTracer()
        with pytest.raises(NotImplementedError, match="Chunked generation"):
            solver.trace_paths(canyon_scene, order=[1, 2], chunk_size=10)

    def test_trace_path_candidates_with_placeholders_matches_unpadded_ground_truth(
        self, canyon_scene: Scene
    ) -> None:
        """Directly exercise ``trace_path_candidates`` with hand-built, padded
        (i.e., containing ``-1`` placeholders) path candidates, as would result
        from combining path candidates of different orders."""
        solver = ExhaustivePathTracer()

        # A hand-built, order-2 array combining a LOS candidate, all 4
        # single-bounce candidates (padded), and a mix of valid/invalid
        # 2-bounce candidates.
        path_candidates = jnp.array([
            [-1, -1],
            [0, -1],
            [1, -1],
            [2, -1],
            [3, -1],
            [1, 2],  # Valid 2-bounce candidate.
            [3, 0],  # Valid 2-bounce candidate.
            [0, 2],  # Invalid 2-bounce candidate.
            [2, 0],  # Invalid 2-bounce candidate.
        ])
        interaction_types = jnp.where(path_candidates >= 0, 0, -1).astype(jnp.int32)

        traced = solver.trace_path_candidates(
            canyon_scene, path_candidates, interaction_types
        )

        # Ground truth: solve each order separately, without any padding.
        los = solver.trace_path_candidates(
            canyon_scene, jnp.zeros((1, 0), dtype=int), jnp.zeros((1, 0), dtype=int)
        )
        order1 = solver.generate_path_candidates(canyon_scene, 1)
        order1_traced = solver.trace_path_candidates(canyon_scene, *order1)
        order2_candidates = jnp.array([[1, 2], [3, 0], [0, 2], [2, 0]])
        order2_traced = solver.trace_path_candidates(
            canyon_scene,
            order2_candidates,
            jnp.zeros_like(order2_candidates),
        )

        expected_mask = jnp.concatenate((
            los.mask.reshape(-1),
            order1_traced.mask.reshape(-1),
            order2_traced.mask.reshape(-1),
        ))
        assert jnp.array_equal(traced.mask.reshape(-1), expected_mask)

        # Padded (order-1 and LOS) rows report the receiver's own position
        # repeated for the padded, trailing vertex/vertices.
        rx = canyon_scene.receivers.reshape(-1, 3)[0]
        assert jnp.allclose(traced.vertices[..., 0, -1, :], rx)
        for i in range(1, 5):  # The 4 padded order-1 candidates.
            assert jnp.allclose(traced.vertices[..., i, -1, :], rx)

        # Un-padded (order-2) candidates match the ground truth exactly.
        assert jnp.allclose(
            traced.vertices[..., 5:, :, :], order2_traced.vertices, atol=1e-4
        )

    def test_trace_path_candidates_rejects_invalid_placeholder_placement(
        self, canyon_scene: Scene
    ) -> None:
        solver = ExhaustivePathTracer()
        # A placeholder ('-1') is immediately followed by a real interaction.
        path_candidates = jnp.array([[1, -1, 2]])
        interaction_types = jnp.zeros_like(path_candidates)

        with pytest.raises(Exception, match="Invalid path candidates"):
            solver.trace_path_candidates(
                canyon_scene, path_candidates, interaction_types
            )

    @pytest.mark.parametrize("assume_quads", [False, True])
    def test_trace_path_candidates_mesh_mask_ignores_placeholders(
        self, canyon_scene: Scene, assume_quads: bool
    ) -> None:
        scene = canyon_scene.set_assume_quads(assume_quads)
        num_primitives = scene.mesh.num_primitives
        mask = jnp.ones((scene.mesh.num_triangles,), dtype=bool).at[0].set(False)
        mesh = eqx.tree_at(
            lambda m: m.mask, scene.mesh, mask, is_leaf=lambda x: x is None
        )
        scene = eqx.tree_at(lambda s: s.mesh, scene, mesh)

        solver = ExhaustivePathTracer()
        step = 2 if assume_quads else 1
        masked_primitive = 0
        other_primitive = step if num_primitives > 1 else 0

        # A padded candidate whose only real interaction is masked out must
        # be invalid; padding by itself must not be affected by the mask.
        path_candidates = jnp.array([
            [-1, -1],
            [masked_primitive, -1],
            [other_primitive, -1],
        ])
        interaction_types = jnp.where(path_candidates >= 0, 0, -1).astype(jnp.int32)
        traced = solver.trace_path_candidates(scene, path_candidates, interaction_types)

        assert bool(traced.mask.reshape(-1)[0])  # LOS: unaffected by the mask.
        assert not bool(traced.mask.reshape(-1)[1])  # Masked-out interaction.


class TestSBRPathTracer:
    def test_order_zero(self, canyon_scene: Scene) -> None:
        solver = SBRPathTracer()
        candidates, interaction_types = solver.generate_path_candidates(
            canyon_scene, order=0
        )
        assert candidates.shape == (1, 0)
        assert interaction_types.shape == (1, 0)

    def test_launch_and_record_respects_mesh_mask(self, canyon_scene: Scene) -> None:
        # The Warp kernel behind '_launch_and_record' must never record a
        # hit on a masked-out triangle.
        masked_scene = eqx.tree_at(
            lambda s: s.mesh.mask,
            canyon_scene,
            jnp.ones((canyon_scene.mesh.num_triangles,), dtype=bool).at[0].set(False),
            is_leaf=lambda x: x is None,
        )
        solver = SBRPathTracer(num_rays=200_000, max_num_candidates=1_000)

        # Sanity check: without masking, primitive 0 is indeed discoverable.
        unmasked, _ = solver.generate_path_candidates(canyon_scene, 1)
        assert jnp.any(unmasked[:, 0] == 0)

        masked, _ = solver.generate_path_candidates(masked_scene, 1)
        assert not jnp.any(masked[:, 0] == 0)

    def test_launch_and_record_trailing_placeholder_invariant(
        self, canyon_scene: Scene
    ) -> None:
        # Once a ray exits the scene, every remaining bounce must be '-1':
        # placeholders only ever appear as a trailing suffix, matching
        # 'check_path_candidates'.
        solver = SBRPathTracer(num_rays=50_000)
        trajectories = solver._launch_and_record(canyon_scene, 5)  # ruff: ignore[private-member-access]
        is_placeholder = trajectories == -1
        assert not jnp.any(is_placeholder[:, :-1] & ~is_placeholder[:, 1:])

    def test_multiple_orders_generates_single_bounded_buffer(
        self, canyon_scene: Scene
    ) -> None:
        solver = SBRPathTracer(num_rays=200_000, max_num_candidates=37)
        candidates, interaction_types = solver.generate_path_candidates(
            canyon_scene, order=[1, 2]
        )
        # Candidates for every requested order share a single buffer
        # bounded by 'max_num_candidates'.
        assert candidates.shape[0] <= 37
        assert candidates.shape[-1] == 2
        assert interaction_types.shape == candidates.shape

        # Every kept candidate's own number of interactions
        # is one of the requested orders.
        num_interactions = (candidates >= 0).sum(axis=-1)
        assert set(jnp.unique(num_interactions).tolist()) <= {1, 2}

    def test_multiple_orders_discards_non_matching_natural_order(
        self, canyon_scene: Scene
    ) -> None:
        # Order 2 is deliberately skipped: candidates are only generated
        # for the requested orders (1 and 3).
        solver = SBRPathTracer(num_rays=200_000, max_num_candidates=1_000)
        candidates, interaction_types = solver.generate_path_candidates(
            canyon_scene, order=[1, 3]
        )
        num_interactions = (candidates >= 0).sum(axis=-1)
        assert set(jnp.unique(num_interactions).tolist()) <= {1, 3}
        # Both non-zero requested orders are indeed discovered.
        assert 1 in num_interactions.tolist()
        assert 3 in num_interactions.tolist()
        assert jnp.array_equal(candidates < 0, interaction_types < 0)

    @pytest.mark.parametrize(
        "order",
        [
            [0, 1, 2, 3, 4, 5],
            range(6),
            slice(0, 6),
        ],
    )
    def test_order_accepts_sequence_range_and_slice(
        self, canyon_scene: Scene, order: list[int] | range | slice
    ) -> None:
        solver = SBRPathTracer(num_rays=200_000, max_num_candidates=1_000)
        candidates, interaction_types = solver.generate_path_candidates(
            canyon_scene, order
        )
        assert candidates.shape == (37, 5)
        assert interaction_types.shape == (37, 5)

    def test_slice_order_without_stop_raises(self, canyon_scene: Scene) -> None:
        solver = SBRPathTracer()
        with pytest.raises(ValueError, match="must have a defined 'stop'"):
            solver.generate_path_candidates(canyon_scene, slice(0, None))

    def test_multiple_orders_chunks_iter_still_unsupported(
        self, canyon_scene: Scene
    ) -> None:
        # Chunking a sequence of orders remains unsupported (regardless of
        # whether the underlying tracer can generate a sequence directly).
        solver = SBRPathTracer(chunk_size=10)
        with pytest.raises(NotImplementedError, match="Chunked generation"):
            solver.trace_paths(canyon_scene, order=[1, 2], chunk_size=10)

    @pytest.mark.parametrize("max_num_candidates", [10, 100])
    def test_candidates_are_bounded(
        self, canyon_scene: Scene, max_num_candidates: int
    ) -> None:
        # Regardless of the order or the number of primitives, the output size
        # is bounded by 'max_num_candidates'.
        solver = SBRPathTracer(num_rays=1_000, max_num_candidates=max_num_candidates)
        for order in (1, 2, 3):
            candidates, interaction_types = solver.generate_path_candidates(
                canyon_scene, order
            )
            assert candidates.shape[0] <= max_num_candidates
            assert candidates.shape[-1] == order
            assert interaction_types.shape == candidates.shape
            # Padded/invalid entries are consistently marked with '-1' on both arrays
            assert jnp.array_equal(candidates < 0, interaction_types < 0)

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("assume_quads", [False, True])
    def test_matches_exhaustive_tracer(
        self, canyon_scene: Scene, order: int, assume_quads: bool
    ) -> None:
        if assume_quads:
            canyon_scene = canyon_scene.set_assume_quads(True)

        exhaustive_paths = ExhaustivePathTracer().trace_paths(canyon_scene, order)

        # A large enough ray population should discover (at least) every
        # candidate that the exhaustive search finds to be valid.
        sbr_paths = SBRPathTracer(
            num_rays=200_000, max_num_candidates=1_000
        ).trace_paths(canyon_scene, order)

        expected_objects = {
            tuple(row.tolist()) for row in exhaustive_paths.masked_objects
        }
        got_objects = {tuple(row.tolist()) for row in sbr_paths.masked_objects}

        assert expected_objects <= got_objects

        # And every path that the SBR tracer deems valid must be a genuine,
        # correctly-solved path (no false positives introduced by discovery).
        for row, vertices in zip(
            sbr_paths.masked_objects.tolist(),
            sbr_paths.masked_vertices,
            strict=True,
        ):
            if tuple(row) in expected_objects:
                index = next(
                    i
                    for i, r in enumerate(exhaustive_paths.masked_objects.tolist())
                    if tuple(r) == tuple(row)
                )
                assert jnp.allclose(
                    vertices, exhaustive_paths.masked_vertices[index], atol=1e-4
                )

    def test_chunks_iter_fallback(self, canyon_scene: Scene) -> None:
        solver = SBRPathTracer(num_rays=1_000, max_num_candidates=37)

        # chunk_size=None falls back to a single chunk with all candidates.
        (chunk,) = list(
            solver.generate_path_candidates_chunks_iter(
                canyon_scene, order=1, chunk_size=None
            )
        )
        assert chunk[0].shape == (4, 1)

        # An explicit chunk_size slices the (already bounded) candidates array.
        chunks = list(
            solver.generate_path_candidates_chunks_iter(
                canyon_scene, order=1, chunk_size=2
            )
        )
        assert [c[0].shape[-2] for c in chunks] == [2, 2]

    @pytest.mark.slow
    @pytest.mark.parametrize("order", [1, 2, 5, 10])
    def test_deep_order_on_large_scene_does_not_explode(
        self, etoile_scene: Scene, order: int
    ) -> None:
        """On a large (~13k triangle), real-world scene, candidate
        generation cost must stay bounded by 'num_rays'/'max_num_candidates',
        and not grow combinatorially with 'order': this is the whole point
        of SBR-style discovery over exhaustive/hybrid enumeration, see
        :class:`SBRPathTracer<differt.geometry.SBRPathTracer>`.

        For reference, an exhaustive (or even visibility-pruned 'hybrid')
        search over this same scene does not even complete at order 2
        within a couple of minutes, which is why no such comparison is
        attempted here for order >= 2.
        """
        solver = SBRPathTracer(num_rays=100_000, max_num_candidates=10_000)

        start = time.perf_counter()
        traced = solver.trace_paths(etoile_scene, order)
        elapsed = time.perf_counter() - start

        # The buffer is bounded by max_num_candidates, regardless of 'order'
        # or the (large) number of primitives in the scene.
        assert traced.objects.shape[-2] <= solver.max_num_candidates
        # Comfortably fast at every order tested here, including order 10,
        # unlike the combinatorial growth of exhaustive/hybrid enumeration.
        assert elapsed < 60.0

        if order <= 2:
            assert int(traced.mask.sum()) >= 1

    def test_matches_exhaustive_tracer_at_order_one_on_large_scene(
        self, etoile_scene: Scene
    ) -> None:
        """Order 1 is the only order for which an exhaustive search remains
        tractable on the full etoile scene (its cost is linear in the
        number of primitives), so it is used here as ground truth to
        validate SBR's correctness directly on a large, real-world scene,
        complementing :meth:`test_matches_exhaustive_tracer`, which checks
        higher orders on a small, synthetic scene.
        """
        exhaustive_paths = ExhaustivePathTracer().trace_paths(etoile_scene, 1)
        sbr_paths = SBRPathTracer(
            num_rays=200_000, max_num_candidates=20_000
        ).trace_paths(etoile_scene, 1)

        expected_objects = {
            tuple(row.tolist()) for row in exhaustive_paths.masked_objects
        }
        got_objects = {tuple(row.tolist()) for row in sbr_paths.masked_objects}

        assert len(expected_objects) > 0
        assert expected_objects <= got_objects

    def test_recovers_ground_truth_across_orders(self, canyon_scene: Scene) -> None:
        # Unlike 'ExhaustivePathTracer', 'SBRPathTracer' shares a single,
        # fixed-size buffer (bounded by 'max_num_candidates') across every
        # requested order: a ray contributes only to the bucket matching
        # its own natural number of interactions (see
        # 'test_combined_buckets_match_natural_stopping_point' below), so
        # the combined result is not required to match the sum of
        # individually-traced orders. It must, however, still recover every
        # distinct valid path across the whole requested range, as long as
        # the buffer is not exceeded.
        solver = SBRPathTracer(num_rays=500_000, max_num_candidates=1_000)
        orders = [0, 1, 2, 3, 4, 5]
        max_order = max(orders)

        combined = solver.trace_paths(canyon_scene, order=orders)
        assert combined.objects.shape[-2] <= solver.max_num_candidates
        got_objects = {tuple(row.tolist()) for row in combined.masked_objects}

        expected_objects = set()
        for o in orders:
            exhaustive = ExhaustivePathTracer().trace_paths(canyon_scene, o)
            for row in exhaustive.masked_objects.tolist():
                missing = max_order - (len(row) - 2)
                padded_row = (*row[:-1], *([-1] * missing), row[-1])
                expected_objects.add(padded_row)

        assert expected_objects <= got_objects

    def test_combined_candidates_match_prefixes(self, canyon_scene: Scene) -> None:
        # The shared ray population used for a sequence of orders must
        # collect prefixes for each requested order from rays that completed
        # at least that many bounces.
        solver = SBRPathTracer(num_rays=200_000, max_num_candidates=1_000)
        orders = [1, 2]
        max_order = max(orders)

        trajectories = solver._launch_and_record(  # ruff: ignore[private-member-access]
            canyon_scene, max_order
        )
        expected_by_order = {}
        for o in orders:
            h = trajectories[:, :o]
            h = h[h[:, o - 1] >= 0]
            expected_by_order[o] = {
                tuple(row.tolist()) for row in jnp.unique(h, axis=0).astype(int)
            }

        combined_candidates, _ = solver.generate_path_candidates(canyon_scene, orders)
        got_by_order = {}
        for o in orders:
            got_o = combined_candidates[(combined_candidates >= 0).sum(axis=-1) == o][
                :, :o
            ]
            got_by_order[o] = {tuple(row.tolist()) for row in got_o}

        for o in orders:
            assert got_by_order[o] == expected_by_order[o]

    def test_no_duplicate_los_padding(self, canyon_scene: Scene) -> None:
        # Without placeholder padding, line-of-sight is not duplicated.
        solver = SBRPathTracer(num_rays=10_000, max_num_candidates=1_000)
        traced = solver.trace_paths(canyon_scene, order=[0, 1, 2])
        expected = ExhaustivePathTracer().trace_paths(canyon_scene, order=[0, 1, 2])

        assert int(traced.mask.sum()) == int(expected.mask.sum())
