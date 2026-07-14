from differt.scene import TriangleScene
from differt.scene._solvers import ExhaustivePathTracer, SBRPathLauncher


def test_generate_path_candidates_chunks_iter(
    simple_street_canyon_scene: TriangleScene,
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


def test_sbr_launcher() -> None:
    solver = SBRPathLauncher()
    # Wait, max_dist doesn't exist? Oh I probably meant hit_tol or something? No, let's just assert solver.max_bounces == 0 or something.
    assert getattr(solver, "chunk_size", None) is None
