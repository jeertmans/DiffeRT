import math
from collections.abc import Sequence
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Bool, PRNGKeyArray

from differt.geometry import TriangleMesh, path_length
from differt.geometry._paths import (
    LaunchPaths,
    Paths,
    SBRPaths,
    TracePaths,
    merge_cell_ids,
)
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


def test_aliases() -> None:
    assert issubclass(Paths, TracePaths)
    assert issubclass(SBRPaths, LaunchPaths)

    with pytest.deprecated_call():
        _ = Paths(jnp.empty((1, 2, 3)), jnp.empty((1, 2), dtype=int))

    with pytest.deprecated_call():
        _ = SBRPaths(
            jnp.empty((1, 2, 3)),
            jnp.empty((1, 2), dtype=int),
            masks=jnp.empty((1, 1), dtype=bool),
        )


def random_paths(
    path_length: int, *batch: int, num_objects: int, with_mask: bool, key: PRNGKeyArray
) -> TracePaths[Bool[Array, "*batch"] | None]:
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

    return TracePaths(vertices, objects, mask)


class TestTracePaths:
    def test_init(self, key: PRNGKeyArray) -> None:
        path_length = 6
        batch = (13, 7)

        paths = random_paths(
            path_length, *batch, num_objects=6, with_mask=True, key=key
        )
        assert paths.shape == batch

        assert paths.mask is not None

        paths = random_paths(
            path_length, *batch, num_objects=6, with_mask=False, key=key
        )
        assert paths.shape == batch

        assert paths.mask is None

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

        paths = scene.trace_paths(path_candidates=path_candidates)

        assert paths.mask is not None
        mask = paths.mask
        paths = eqx.tree_at(lambda p: p.mask, paths, jnp.ones_like(paths.mask))

        got = paths.mask_duplicate_objects()

        chex.assert_trees_all_equal(got.num_valid_paths, 3)

        paths = eqx.tree_at(lambda p: p.mask, paths, None)

        got = paths.mask_duplicate_objects()

        chex.assert_trees_all_equal(got.num_valid_paths, 3)

        paths = eqx.tree_at(
            lambda p: p.mask,
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

        paths = scene.trace_paths(path_candidates=path_candidates)

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
            lambda p: p.mask,
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
            match=r"Cannot create multipath cells from non-existing mask!",
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

            assert isinstance(path, TracePaths)
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

        expected = path_length(paths.vertices).sum(axis=axis, where=paths.mask)

        got = paths.reduce(path_length, axis=axis)

        chex.assert_shape(got, expected_shape)
        chex.assert_trees_all_equal(got, expected)

    @pytest.mark.parametrize("backend", [plotly, matplotlib, vispy])
    def test_plot(self, backend: str, key: PRNGKeyArray) -> None:
        paths = random_paths(3, 4, 5, num_objects=30, with_mask=True, key=key)

        _ = paths.plot(backend=backend)


class TestLaunchPaths:
    def test_init(self, key: PRNGKeyArray) -> None:
        key_paths, key_masks = jax.random.split(key, 2)

        path_length = 5
        batch = (30, 10)

        paths = random_paths(
            path_length, *batch, num_objects=30, with_mask=True, key=key_paths
        )
        assert paths.shape == batch

        masks = jax.random.uniform(key_masks, (*batch, path_length - 1)) > 0.5

        launch_paths = LaunchPaths(paths.vertices, paths.objects, masks=masks)

        assert launch_paths.vertices.shape == (*batch, path_length, 3)
        assert launch_paths.objects.shape == (*batch, path_length)
        assert launch_paths.mask is not None
        assert launch_paths.mask.shape == batch
        assert launch_paths.masks.shape == (*batch, path_length - 1)

        chex.assert_trees_all_equal(launch_paths.masks[..., -1], launch_paths.mask)

    def test_get_paths(self, key: PRNGKeyArray) -> None:
        key_paths, key_masks = jax.random.split(key, 2)

        path_length = 4
        batch = (50,)

        paths = random_paths(
            path_length, *batch, num_objects=30, with_mask=False, key=key_paths
        )
        assert paths.shape == batch

        masks = jax.random.uniform(key_masks, (*batch, path_length - 1)) > 0.5

        launch_paths = LaunchPaths(paths.vertices, paths.objects, masks=masks)
        del paths

        for i in range(path_length - 1):
            paths = launch_paths.get_paths(i)
            chex.assert_trees_all_equal(paths.mask, launch_paths.masks[..., i])

        for i in [-1, path_length - 1]:
            with pytest.raises(
                ValueError,
                match=f"Paths order must be strictly between 0 and {path_length - 2}",
            ):
                _ = launch_paths.get_paths(i)

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

        launch_paths = LaunchPaths(paths.vertices, paths.objects, masks=masks)
        _ = launch_paths.plot(backend=backend)

    def test_additional_methods(self, key: PRNGKeyArray) -> None:
        key_paths, key_masks = jax.random.split(key, 2)

        path_length = 4
        batch = (2, 3)

        paths = random_paths(
            path_length, *batch, num_objects=10, with_mask=False, key=key_paths
        )
        masks = jax.random.uniform(key_masks, (*batch, path_length - 1)) > 0.5

        launch_paths = LaunchPaths(paths.vertices, paths.objects, masks=masks)

        # Test shape, path_length, order
        assert launch_paths.shape == batch
        assert launch_paths.path_length == path_length
        assert launch_paths.order == path_length - 2

        # Test reshape
        reshaped = launch_paths.reshape(6)
        assert reshaped.shape == (6,)
        assert reshaped.vertices.shape == (6, path_length, 3)

        # Test squeeze
        squeezed_launch_paths = LaunchPaths(
            paths.vertices.reshape(2, 1, 3, path_length, 3),
            paths.objects.reshape(2, 1, 3, path_length),
            masks=masks.reshape(2, 1, 3, path_length - 1),
        )
        squeezed = squeezed_launch_paths.squeeze(axis=1)
        assert squeezed.shape == (2, 3)

        # Test __iter__
        got_iter = list(launch_paths)
        assert len(got_iter) == launch_paths.masked().num_valid_paths

        # Test masked, masked_vertices, masked_objects
        masked_paths = launch_paths.masked()
        assert isinstance(masked_paths, TracePaths)
        chex.assert_trees_all_equal(launch_paths.masked_vertices, masked_paths.vertices)
        chex.assert_trees_all_equal(
            launch_paths.masked_objects,
            launch_paths.get_paths(launch_paths.order).masked_objects,
        )
