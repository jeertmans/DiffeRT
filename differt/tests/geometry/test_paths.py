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
            match=r"Cannot create multipath cells from non-existing mask \(or confidence matrix\)!",
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

    def test_fuse_planes_basic(self, key: PRNGKeyArray) -> None:
        """Test basic fuse_planes functionality with simple case."""
        key_paths, key_masks = jax.random.split(key, 2)

        path_length = 4
        num_tx = 2
        num_planes = 3
        num_rays = 10

        # Create paths with plane dimension [num_tx, num_planes, num_rays, path_length, 3]
        vertices = jax.random.uniform(
            key_paths, (num_tx, num_planes, num_rays, path_length, 3)
        )
        objects = jax.random.randint(
            key_paths, (num_tx, num_planes, num_rays, path_length), minval=0, maxval=5
        )

        # Create masks [num_tx, num_planes, num_rays, path_length-1]
        # Make sure each ray has at most one plane with True mask (non-overlapping assumption)
        masks = jnp.zeros((num_tx, num_planes, num_rays, path_length - 1), dtype=bool)

        # For each ray, randomly select one plane to be True
        for tx_idx in range(num_tx):
            for ray_idx in range(num_rays):
                plane_idx = jax.random.randint(
                    jax.random.fold_in(key_masks, tx_idx * num_rays + ray_idx),
                    (),
                    minval=0,
                    maxval=num_planes,
                )
                masks = masks.at[tx_idx, plane_idx, ray_idx, :].set(True)

        sbr_paths = SBRPaths(vertices, objects, masks=masks)

        # Fuse the planes
        fused_paths = sbr_paths.fuse_planes()

        # Check shapes
        assert fused_paths.vertices.shape == (num_tx, num_rays, path_length, 3)
        assert fused_paths.objects.shape == (num_tx, num_rays, path_length)
        assert fused_paths.masks.shape == (num_tx, num_rays, path_length - 1)

    def test_fuse_planes_no_batch(self, key: PRNGKeyArray) -> None:
        """Test fuse_planes with minimal batch dimensions."""
        key_paths, _key_masks = jax.random.split(key, 2)

        path_length = 3
        num_planes = 2
        num_rays = 5

        # Create paths without tx batch dimension [num_planes, num_rays, path_length, 3]
        # Note: Need at least one batch dimension for the structure
        vertices = jax.random.uniform(
            key_paths, (1, num_planes, num_rays, path_length, 3)
        )
        objects = jax.random.randint(
            key_paths, (1, num_planes, num_rays, path_length), minval=0, maxval=3
        )

        # Create masks where first plane is always selected
        masks = jnp.zeros((1, num_planes, num_rays, path_length - 1), dtype=bool)
        masks = masks.at[0, 0, :, :].set(True)

        sbr_paths = SBRPaths(vertices, objects, masks=masks)
        fused_paths = sbr_paths.fuse_planes()

        assert fused_paths.vertices.shape == (1, num_rays, path_length, 3)
        # Verify that the first plane's data is selected
        chex.assert_trees_all_close(
            fused_paths.vertices, vertices[0, 0, :, :, :][None, :, :, :]
        )

    def test_fuse_planes_selection_logic(self) -> None:
        """Test that fuse_planes correctly selects the first plane with True mask."""
        path_length = 3
        num_tx = 1
        num_planes = 4
        num_rays = 3

        # Manually create vertices with distinct values per plane
        vertices = jnp.zeros((num_tx, num_planes, num_rays, path_length, 3))
        for plane_idx in range(num_planes):
            # Set a unique value for each plane (plane_idx + 1)
            vertices = vertices.at[0, plane_idx, :, :, 0].set(float(plane_idx + 1))

        objects = jnp.zeros(
            (num_tx, num_planes, num_rays, path_length), dtype=jnp.int32
        )

        # Create masks where different rays select different planes
        masks = jnp.zeros((num_tx, num_planes, num_rays, path_length - 1), dtype=bool)

        # Ray 0: select plane 1 (index 1)
        masks = masks.at[0, 1, 0, :].set(True)
        # Ray 1: select plane 2 (index 2)
        masks = masks.at[0, 2, 1, :].set(True)
        # Ray 2: select plane 0 (index 0)
        masks = masks.at[0, 0, 2, :].set(True)

        sbr_paths = SBRPaths(vertices, objects, masks=masks)
        fused_paths = sbr_paths.fuse_planes()

        # Verify correct plane selection
        # Ray 0 should have plane 1's data (value 2.0 in first coordinate)
        assert jnp.allclose(fused_paths.vertices[0, 0, :, 0], 2.0)
        # Ray 1 should have plane 2's data (value 3.0 in first coordinate)
        assert jnp.allclose(fused_paths.vertices[0, 1, :, 0], 3.0)
        # Ray 2 should have plane 0's data (value 1.0 in first coordinate)
        assert jnp.allclose(fused_paths.vertices[0, 2, :, 0], 1.0)

    def test_fuse_planes_no_interceptions(self, key: PRNGKeyArray) -> None:
        """Test fuse_planes when some rays have no interceptions."""
        path_length = 3
        num_tx = 1
        num_planes = 2
        num_rays = 4

        vertices = jax.random.uniform(
            key, (num_tx, num_planes, num_rays, path_length, 3)
        )
        objects = jax.random.randint(
            key, (num_tx, num_planes, num_rays, path_length), minval=0, maxval=3
        )

        # Create masks where some rays have no True values
        masks = jnp.zeros((num_tx, num_planes, num_rays, path_length - 1), dtype=bool)
        # Only rays 0 and 2 have interceptions
        masks = masks.at[0, 0, 0, :].set(True)
        masks = masks.at[0, 1, 2, :].set(True)
        # Rays 1 and 3 have no interceptions (all False)

        sbr_paths = SBRPaths(vertices, objects, masks=masks)
        fused_paths = sbr_paths.fuse_planes()

        # Check that rays without interceptions have False masks
        assert jnp.all(~fused_paths.masks[0, 1, :])
        assert jnp.all(~fused_paths.masks[0, 3, :])
        # Rays with interceptions should have True masks
        assert jnp.all(fused_paths.masks[0, 0, :])
        assert jnp.all(fused_paths.masks[0, 2, :])

    def test_fuse_planes_error_on_insufficient_dims(self) -> None:
        """Test that fuse_planes raises error for insufficient dimensions."""
        # Create paths with too few dimensions
        vertices = jnp.ones((10, 3, 3))  # Only 3 dimensions
        objects = jnp.zeros((10, 3), dtype=jnp.int32)
        masks = jnp.ones((10, 2), dtype=bool)

        sbr_paths = SBRPaths(vertices, objects, masks=masks)

        with pytest.raises(
            ValueError, match="vertices shape must have at least 4 dimensions"
        ):
            sbr_paths.fuse_planes()
