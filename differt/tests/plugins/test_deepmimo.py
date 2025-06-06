from dataclasses import asdict
from itertools import chain

import chex
import jax
import numpy as np
import pytest
from jaxtyping import PRNGKeyArray

from differt.em import materials
from differt.geometry import TriangleMesh
from differt.plugins import deepmimo
from differt.scene import TriangleScene
from differt.utils import sample_points_in_bounding_box


def test_export(key: PRNGKeyArray) -> None:
    mesh = TriangleMesh.box(with_top=True, with_bottom=True)
    key_tx, key_rx = jax.random.split(key, 2)

    transmitters = sample_points_in_bounding_box(mesh.bounding_box, (1, 2), key=key_tx)
    receivers = sample_points_in_bounding_box(mesh.bounding_box, (4,), key=key_rx)
    num_tx = transmitters[..., 0].size
    num_rx = receivers[..., 0].size
    scene = TriangleScene(mesh=mesh, transmitters=transmitters, receivers=receivers)

    with pytest.raises(
        ValueError, match="Scene must contain information about face materials"
    ):
        deepmimo.export(
            paths=scene.compute_paths(order=0), scene=scene, frequency=2.4e9
        )

    mesh = mesh.set_materials("itu_concrete")
    scene = TriangleScene(mesh=mesh, transmitters=transmitters, receivers=receivers)

    frequency = 2.4e9  # 2.4 GHz
    for order in [0, 1, 2]:
        paths = scene.compute_paths(order=order)
        dm = deepmimo.export(paths=paths, scene=scene, frequency=frequency)
        assert dm.num_tx == num_tx
        assert dm.num_rx == num_rx
        assert dm.num_paths == paths.vertices.shape[-3]
        paths_iter = scene.compute_paths(order=order, chunk_size=100)
        dm = deepmimo.export(
            paths=paths_iter,
            scene=scene,
            radio_materials=materials,
            frequency=frequency,
        )
        assert dm.num_tx == num_tx
        assert dm.num_rx == num_rx
        assert dm.num_paths == paths.vertices.shape[-3]

    paths_iter = (scene.compute_paths(order=order) for order in [0, 1, 2])
    dm = deepmimo.export(paths=paths_iter, scene=scene, frequency=frequency)
    assert dm.num_tx == num_tx
    assert dm.num_rx == num_rx
    num_paths = dm.num_paths
    paths_iter_iter = chain.from_iterable(
        scene.compute_paths(order=order, chunk_size=100) for order in [0, 1, 2]
    )
    dm = deepmimo.export(paths=paths_iter_iter, scene=scene, frequency=frequency)
    assert dm.num_tx == num_tx
    assert dm.num_rx == num_rx
    assert dm.num_paths == num_paths

    assert len(dm.asdict()) == len(asdict(dm))

    assert all(isinstance(arr, jax.Array) for arr in asdict(dm).values())
    assert all(isinstance(arr, np.ndarray) for arr in asdict(dm.numpy()).values())

    # Check round-trip conversion
    chex.assert_trees_all_equal(
        dm.numpy().jax(),
        dm,
    )
