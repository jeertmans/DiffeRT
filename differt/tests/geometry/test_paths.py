import math

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PRNGKeyArray

from differt.geometry.paths import Paths


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
    @pytest.mark.parametrize("path_length", [1, 3, 5])
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
        got = paths.masked_vertices

        num_paths = (
            int(paths.mask.sum()) if paths.mask is not None else math.prod(batch)
        )

        assert got.size == num_paths * path_length * 3

        got = paths.masked_objects

        assert got.size == num_paths * path_length

    @pytest.mark.parametrize("path_length", [1, 2, 3])
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

    def test_iter(self, key: PRNGKeyArray) -> None:
        paths = random_paths(6, 3, 2, num_objects=20, with_mask=True, key=key)

        got = 0
        for path in paths:
            got += 1

            assert isinstance(path, Paths)
            assert path.num_valid_paths == 1

        assert got == paths.num_valid_paths

    @pytest.mark.parametrize("backend", ["plotly", "matplotlib", "vispy"])
    def test_plot(self, backend: str, key: PRNGKeyArray) -> None:
        paths = random_paths(3, 4, 5, num_objects=30, with_mask=True, key=key)

        _ = paths.plot(backend=backend)
