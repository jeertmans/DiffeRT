import math
from collections.abc import Sequence
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PRNGKeyArray

from differt.geometry import TriangleMesh, path_lengths
from differt.geometry._paths import Paths, SBRPaths, merge_cell_ids
from differt.scene import TriangleScene

from ..plotting.params import matplotlib, plotly, vispy


def test_merge_cell_ids() -> None:
    cell_a_ids = jnp.array(
        [4, 0, 2, 0, 4],
    )
    cell_b_ids = jnp.array(
        [1, 3, 7, 3, 1],
    )
    expected = jnp.array([0, 1, 2, 1, 0])
    got = merge_cell_ids(cell_a_ids, cell_b_ids)

    chex.assert_trees_all_equal(got, expected)


def random_paths(
    path_length: int, *batch: int, num_objects: int, with_mask: bool, key: PRNGKeyArray
) -> Paths:
    if with_mask:
        key_vertices, key_objects, key_mask = jax.random.split(key, 3)
        mask = jax.random.uniform(key_mask, batch) > 0.5
    else:
        key_vertices, key_objects = jax.random.split(key, 2)
        mask = None

    vertices = jax.random.uniform(key_vertices, (*batch, path_length, 3))
    objects = jax.random.randint(
        key_objects, (*batch, path_length), minval=0, maxval=num_objects
    )

    return Paths(vertices, objects, mask)


class TestPaths:
    def test_init(self, key: PRNGKeyArray) -> None:
        path_length = 6
        batch = (13, 7)

        paths = random_paths(
            path_length, *batch, num_objects=6, with_mask=True, key=key
        )
        assert paths.shape == batch

        assert paths.mask is not None

        with pytest.warns(
            UserWarning, match="Setting both 'mask' and 'confidence' arguments"
        ):
            paths = Paths(
                paths.vertices,
                paths.objects,
                mask=paths.mask,
                confidence=paths.mask.astype(float),
            )

    @pytest.mark.parametrize("with_mask", [False, True])
    @pytest.mark.parametrize(
        ("batch", "axis", "expectation"),
        [
            ((), None, does_not_raise()),
            (
                (),
                -1,
                pytest.raises(
                    ValueError, match="Cannot squeeze a 0-dimensional batch!"
                ),
            ),
            ((1,), -1, does_not_raise()),
            ((1,), 0, does_not_raise()),
            ((10, 1), -1, does_not_raise()),
            ((10, 1), 1, does_not_raise()),
            ((1, 1), (0, 1), does_not_raise()),
            (
                (1,),
                1,
                pytest.raises(
                    ValueError, match="One of the provided axes is out-of-bounds!"
                ),
            ),
        ],
    )
    def test_squeeze(
        self,
        with_mask: bool,
        batch: tuple[int, ...],
        axis: int | tuple[int, ...] | None,
        expectation: AbstractContextManager[Exception],
        key: PRNGKeyArray,
    ) -> None:
        path_length = 10
        num_objects = 30
        with expectation:
            _ = random_paths(
                path_length,
                *batch,
                num_objects=num_objects,
                with_mask=with_mask,
                key=key,
            ).squeeze(axis=axis)

    def test_mask_duplicate_objects(self, key: PRNGKeyArray) -> None:
        mesh = TriangleMesh.box()  # 6 objects
        # 6 path candidates, only 3 are unique
        path_candidates = jnp.array([
            [0, 1, 2],
            [1, 0, 2],
            [0, 1, 2],
            [0, 1, 2],
            [2, 3, 4],
            [1, 0, 2],
        ])

        # 1 - One TX, one RX, no batch dimension

        key_rx, key_tx = jax.random.split(key, 2)

        scene = TriangleScene(
            transmitters=jax.random.normal(key_tx, (3,)),
            receivers=jax.random.normal(key_rx, (3,)),
            mesh=mesh,
        )

        paths = scene.compute_paths(path_candidates=path_candidates)

        assert paths.mask is not None
        mask = paths.mask
        paths = eqx.tree_at(lambda p: p.mask, paths, jnp.ones_like(paths.mask))

        got = paths.mask_duplicate_objects()

        chex.assert_trees_all_equal(got.num_valid_paths, 3)

        paths = eqx.tree_at(lambda p: p.mask, paths, None)

        got = paths.mask_duplicate_objects()

        chex.assert_trees_all_equal(got.num_valid_paths, 3)

        paths = eqx.tree_at(
            lambda p: p.confidence,
            paths,
            jnp.ones_like(mask, dtype=float),
            is_leaf=lambda x: x is None,
        )

        got = paths.mask_duplicate_objects()

        chex.assert_trees_all_equal(got.num_valid_paths, 3)

        with pytest.raises(
            ValueError,
            match="The provided axis -2 is out-of-bounds for batch of dimensions 1!",
        ):
            _ = paths.mask_duplicate_objects(axis=-2)

        # 2 - Many TXs, many RXs, multiple batch dimensions

        scene = scene.with_transmitters_grid(2, 1).with_receivers_grid(4, 3)

        batch_size = scene.num_transmitters * scene.num_receivers

        paths = scene.compute_paths(path_candidates=path_candidates)

        assert paths.mask is not None
        mask = paths.mask
        chex.assert_shape(paths.mask, (1, 2, 3, 4, path_candidates.shape[0]))
        paths = eqx.tree_at(lambda p: p.mask, paths, jnp.ones_like(paths.mask))

        got = paths.mask_duplicate_objects()

        chex.assert_shape(got.mask, (1, 2, 3, 4, path_candidates.shape[0]))
        chex.assert_trees_all_equal(got.num_valid_paths, 3 * batch_size)

        paths = eqx.tree_at(lambda p: p.mask, paths, None)

        got = paths.mask_duplicate_objects()

        chex.assert_shape(got.mask, (1, 2, 3, 4, path_candidates.shape[0]))
        chex.assert_trees_all_equal(got.num_valid_paths, 3 * batch_size)

        paths = eqx.tree_at(
            lambda p: (p.vertices, p.objects),
            paths,
            (jnp.swapaxes(paths.vertices, 0, -3), jnp.swapaxes(paths.objects, 0, -2)),
        )

        got = paths.mask_duplicate_objects(axis=0)

        chex.assert_shape(got.mask, (path_candidates.shape[0], 2, 3, 4, 1))
        chex.assert_trees_all_equal(got.num_valid_paths, 3 * batch_size)

        paths = eqx.tree_at(
            lambda p: p.confidence,
            paths,
            jnp.ones((path_candidates.shape[0], 2, 3, 4, 1), dtype=float),
            is_leaf=lambda x: x is None,
        )

        got = paths.mask_duplicate_objects(axis=0)

        chex.assert_trees_all_equal(got.num_valid_paths, 3 * batch_size)

    @pytest.mark.parametrize("path_length", [3, 5])
    @pytest.mark.parametrize("batch", [(), (1,), (1, 2, 3, 4)])
    @pytest.mark.parametrize("num_objects", [1, 10])
    @pytest.mark.parametrize("with_mask", [False, True])
    def test_masked_vertices_and_objects(
        self,
        path_length: int,
        batch: tuple[int, ...],
        num_objects: int,
        with_mask: bool,
        key: PRNGKeyArray,
    ) -> None:
        paths = random_paths(
            path_length, *batch, num_objects=num_objects, with_mask=with_mask, key=key
        )
        assert paths.shape == batch
        got = paths.masked_vertices

        assert paths.path_length == path_length
        assert paths.order == path_length - 2

        num_paths = (
            int(paths.mask.sum()) if paths.mask is not None else math.prod(batch)
        )

        assert got.size == num_paths * path_length * 3

        got = paths.masked_objects

        assert got.size == num_paths * path_length

    @pytest.mark.parametrize("path_length", [2, 3])
    @pytest.mark.parametrize("batch", [(45,), (1, 10, 3, 4)])
    @pytest.mark.parametrize("num_objects", [2, 3])
    def test_group_by_objects(
        self,
        path_length: int,
        batch: tuple[int, ...],
        num_objects: int,
        key: PRNGKeyArray,
    ) -> None:
        paths = random_paths(
            path_length, *batch, num_objects=num_objects, with_mask=False, key=key
        )
        assert paths.shape == batch
        got = paths.group_by_objects()

        assert got.shape == batch

        got = got.reshape(-1)
        objects = paths.objects.reshape((-1, path_length))

        at_least_one_test = False

        for object_index in range(num_objects):
            indices = jnp.argwhere(got == object_index)

            same_objects = jnp.take(objects, indices, axis=0)

            if same_objects.shape[0] > 1:
                at_least_one_test = True
                chex.assert_trees_all_equal(*same_objects)

        assert at_least_one_test, "This test is useless, please remove."

    def test_multipath_cells(self, key: PRNGKeyArray) -> None:
        with pytest.raises(
            ValueError,
            match=r"Cannot create multiplath cells from non-existing mask \(or confidence matrix\)!",
        ):
            _ = random_paths(
                6, 3, 2, num_objects=20, with_mask=False, key=key
            ).multipath_cells()

        paths = random_paths(3, 6, 2, num_objects=1, with_mask=True, key=key)
        paths = eqx.tree_at(
            lambda p: p.mask,
            paths,
            jnp.array([
                [True, False],
                [True, True],
                [True, False],
                [False, False],
                [False, True],
                [False, False],
            ]),
        )

        got = paths.multipath_cells()
        expected = jnp.array([0, 1, 0, 3, 4, 3])

        chex.assert_trees_all_equal(got, expected)

    def test_iter(self, key: PRNGKeyArray) -> None:
        paths = random_paths(6, 3, 2, num_objects=20, with_mask=True, key=key)

        got = 0
        for path in paths:
            got += 1

            assert isinstance(path, Paths)
            assert path.num_valid_paths == 1

        assert got == paths.num_valid_paths

    @pytest.mark.parametrize(
        ("batch", "axis", "expected_shape"),
        [
            ((), None, ()),
            ((10,), None, ()),
            ((5, 20), -1, (5,)),
            ((15, 20), (0, 1), ()),
        ],
    )
    def test_reduce(
        self,
        batch: tuple[int, ...],
        axis: Sequence[int] | int | None,
        expected_shape: tuple[int, ...],
        key: PRNGKeyArray,
    ) -> None:
        paths = random_paths(4, *batch, num_objects=3, with_mask=True, key=key)
        assert paths.shape == batch

        expected = path_lengths(paths.vertices).sum(axis=axis, where=paths.mask)

        got = paths.reduce(path_lengths, axis=axis)

        chex.assert_shape(got, expected_shape)
        chex.assert_trees_all_equal(got, expected)

    @pytest.mark.parametrize("backend", [plotly, matplotlib, vispy])
    def test_plot(self, backend: str, key: PRNGKeyArray) -> None:
        paths = random_paths(3, 4, 5, num_objects=30, with_mask=True, key=key)

        _ = paths.plot(backend=backend)


class TestSBRPaths:
    def test_init(self, key: PRNGKeyArray) -> None:
        key_paths, key_masks = jax.random.split(key, 2)

        path_length = 5
        batch = (30, 10)

        paths = random_paths(
            path_length, *batch, num_objects=30, with_mask=True, key=key_paths
        )
        assert paths.shape == batch

        assert paths.mask is not None
        mask = paths.mask

        masks = jax.random.uniform(key_masks, (*batch, path_length - 1)) > 0.5

        sbr_paths = SBRPaths(paths.vertices, paths.objects, masks=masks)

        assert sbr_paths.vertices.shape == (*batch, path_length, 3)
        assert sbr_paths.objects.shape == (*batch, path_length)
        assert sbr_paths.mask is not None
        assert sbr_paths.mask.shape == batch
        assert sbr_paths.masks.shape == (*batch, path_length - 1)

        chex.assert_trees_all_equal(sbr_paths.masks[..., -1], sbr_paths.mask)

        with pytest.warns(
            UserWarning, match="Setting 'mask' argument is ignored for this class"
        ):
            sbr_paths = SBRPaths(paths.vertices, paths.objects, mask=mask, masks=masks)

        with pytest.raises(AssertionError):  # Check that mask param is not used
            chex.assert_trees_all_equal(sbr_paths.mask, mask)

        chex.assert_trees_all_equal(sbr_paths.masks[..., -1], sbr_paths.mask)

    def test_get_paths(self, key: PRNGKeyArray) -> None:
        key_paths, key_masks = jax.random.split(key, 2)

        path_length = 4
        batch = (50,)

        paths = random_paths(
            path_length, *batch, num_objects=30, with_mask=False, key=key_paths
        )
        assert paths.shape == batch

        masks = jax.random.uniform(key_masks, (*batch, path_length - 1)) > 0.5

        sbr_paths = SBRPaths(paths.vertices, paths.objects, masks=masks)
        del paths

        for i in range(path_length - 1):
            paths = sbr_paths.get_paths(i)
            chex.assert_trees_all_equal(paths.mask, sbr_paths.masks[..., i])

        for i in [-1, path_length - 1]:
            with pytest.raises(
                ValueError,
                match=f"Paths order must be strictly between 0 and {path_length - 2}",
            ):
                _ = sbr_paths.get_paths(i)

    @pytest.mark.parametrize("backend", [plotly, matplotlib, vispy])
    def test_plot(self, backend: str, key: PRNGKeyArray) -> None:
        key_paths, key_masks = jax.random.split(key, 2)

        path_length = 3
        batch = (
            3,
            50,
        )

        paths = random_paths(
            path_length, *batch, num_objects=30, with_mask=False, key=key_paths
        )
        assert paths.shape == batch

        masks = jax.random.uniform(key_masks, (*batch, path_length - 1)) > 0.5

        sbr_paths = SBRPaths(paths.vertices, paths.objects, masks=masks)
        _ = sbr_paths.plot(backend=backend)
