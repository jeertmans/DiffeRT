from collections.abc import Iterator
from dataclasses import asdict
from itertools import chain

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import PRNGKeyArray

from differt.em import materials
from differt.geometry import Paths, TriangleMesh
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


@pytest.mark.slow
def test_match_sionna_on_simple_street_canyon() -> None:
    mi = pytest.importorskip("mitsuba")
    mi.set_variant("llvm_ad_mono_polarized")
    sionna = pytest.importorskip("sionna")
    file = sionna.rt.scene.simple_street_canyon

    sionna_scene = sionna.rt.load_scene(file)
    differt_scene = TriangleScene.load_xml(file)

    sionna_scene.tx_array = sionna.rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="cross",
    )

    sionna_scene.rx_array = sionna.rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="cross",
    )

    tx = sionna.rt.Transmitter(name="tx", position=[-33.0, 0.0, 32.0])

    sionna_scene.add(tx)

    rx = sionna.rt.Receiver(name="rx", position=[20.0, 0.0, 2.0], orientation=[0, 0, 0])

    sionna_scene.add(rx)

    tx.look_at(rx)

    differt_scene = eqx.tree_at(
        lambda s: s.transmitters,
        differt_scene,
        replace=tx.position.jax().reshape(3),
    )

    differt_scene = eqx.tree_at(
        lambda s: s.receivers,
        differt_scene,
        replace=rx.position.jax().reshape(3),
    )

    max_order = 3

    sionna_solver = sionna.rt.PathSolver()
    sionna_paths = sionna_solver(sionna_scene, max_depth=max_order, refraction=False)
    sionna_path_primitives = sionna_paths.primitives.jax()
    where_invalid_primitives = (
        sionna_paths.primitives == sionna.rt.constants.INVALID_PRIMITIVE
    ).jax()
    sionna_path_primitives = jnp.where(
        where_invalid_primitives, 0, sionna_path_primitives
    ).astype(jnp.int32)  # type: ignore[reportAttributeAccessIssue]
    sionna_path_primitives = jnp.where(
        where_invalid_primitives, -1, sionna_path_primitives
    )
    sionna_path_primitives = jnp.moveaxis(sionna_path_primitives, 0, -1)

    def paths_iter(max_depth: int) -> Iterator[Paths]:
        for order in range(max_depth + 1):
            select = (
                sionna_paths.interactions != sionna.rt.InteractionType.NONE
            ).jax().sum(axis=0) == order
            path_candidates = sionna_path_primitives[select, :order]
            p = differt_scene.compute_paths(path_candidates=path_candidates)
            yield p

    dm = deepmimo.export(
        paths=paths_iter(max_depth=max_order),
        scene=differt_scene,
        frequency=sionna_scene.frequency.jax().item(0),
        include_primitives=True,
    )
    dm = dm._sort(sionna_path_primitives)  # noqa: SLF001

    assert dm.num_tx == sionna_paths.num_tx == 1
    assert dm.num_rx == sionna_paths.num_rx == 1

    chex.assert_trees_all_equal(
        dm.primitives,
        sionna_path_primitives,
    )

    # TODO: check why we get more than one antenna per transmitter/receiver

    chex.assert_trees_all_equal(
        jnp.deg2rad(dm.phase),
        jnp.arctan2(sionna_paths.a[1].jax(), sionna_paths.a[0].jax()).squeeze(
            axis=(0, 2)
        ),
    )
