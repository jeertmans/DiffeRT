# ruff: noqa: ERA001
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
from differt.geometry import TriangleMesh
from differt.plugins import deepmimo
from differt.rt import triangles_visible_from_vertices
from differt.scene import TriangleScene
from differt.utils import sample_points_in_bounding_box
from differt_core.rt import CompleteGraph, DiGraph


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
    assert dm.primitives is None

    assert len(dm.asdict()) == len(asdict(dm))

    assert all(
        isinstance(arr, jax.Array) if arr is not None else True
        for arr in asdict(dm).values()
    ), f"{dm!r}"
    assert all(
        isinstance(arr, np.ndarray) if arr is not None else True
        for arr in asdict(dm.numpy()).values()
    ), f"{dm.numpy()!r}"

    # Check round-trip conversion
    chex.assert_trees_all_equal(
        dm.numpy().jax(),
        dm,
    )


@pytest.mark.slow
def test_match_sionna_on_simple_street_canyon() -> None:
    mi = pytest.importorskip("mitsuba", reason="mitsuba not installed")
    try:
        mi.set_variant("llvm_ad_mono_polarized")
    except AttributeError:
        pytest.skip("Mitsuba variant 'llvm_ad_mono_polarized' not available")
    sionna = pytest.importorskip("sionna", reason="sionna not installed")
    file = sionna.rt.scene.simple_street_canyon

    sionna_scene = sionna.rt.load_scene(file)
    differt_scene = TriangleScene.load_xml(file).set_assume_quads()  # Faster RT

    sionna_scene.tx_array = sionna.rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )

    sionna_scene.rx_array = sionna.rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )

    tx = sionna.rt.Transmitter(name="tx", position=[-33.0, 0.0, 32.0])

    sionna_scene.add(tx)

    rx = sionna.rt.Receiver(name="rx", position=[20.0, 0.0, 2.0])

    sionna_scene.add(rx)

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

    k = 2 if differt_scene.mesh.assume_quads else 1

    triangles_visible_from_tx = (
        triangles_visible_from_vertices(
            differt_scene.transmitters,
            differt_scene.mesh.triangle_vertices,
            differt_scene.mesh.mask,
        )
        .reshape(-1, k)
        .any(axis=-1)
    )

    triangles_visible_from_rx = (
        triangles_visible_from_vertices(
            differt_scene.receivers,
            differt_scene.mesh.triangle_vertices,
            differt_scene.mesh.mask,
        )
        .reshape(-1, k)
        .any(axis=-1)
    )

    di_graph = DiGraph.from_complete_graph(
        CompleteGraph(differt_scene.mesh.num_objects)
    )
    from_, to = di_graph.insert_from_and_to_nodes(
        from_adjacency=np.asarray(triangles_visible_from_tx),
        to_adjacency=np.asarray(triangles_visible_from_rx),
    )

    max_order = 3

    sionna_solver = sionna.rt.PathSolver()
    sionna_paths = sionna_solver(sionna_scene, max_depth=max_order, refraction=False)

    dm = deepmimo.export(
        paths=(
            differt_scene.compute_paths(
                path_candidates=jnp.asarray(k * path_candidates, dtype=int)
            ).masked()
            for order in range(max_order + 1)
            for path_candidates in di_graph.all_paths_array_chunks(
                from_=from_,
                to=to,
                depth=order + 2,
                include_from_and_to=False,
                chunk_size=100_000,
            )
        ),
        scene=differt_scene,
        frequency=sionna_scene.frequency.jax().item(0),
        include_primitives=True,
    )

    # Greedily sort the paths to match Sionna's order
    dm = dm._sort(sionna_paths)  # noqa: SLF001

    assert dm.num_tx == sionna_paths.num_tx == 1
    assert dm.num_rx == sionna_paths.num_rx == 1

    chex.assert_trees_all_equal(
        dm.inter + 1,  # +1 to match Sionna's numbering
        jnp.moveaxis(sionna_paths.interactions.jax(), 0, -1),
    )

    chex.assert_trees_all_equal(
        dm.mask,
        True,  # All paths are valid in this case  # noqa: FBT003
    )

    chex.assert_trees_all_close(
        dm.inter_pos,
        jnp.moveaxis(sionna_paths.vertices.jax(), 0, -2),
        atol=1e-3,
    )

    chex.assert_trees_all_close(
        dm.delay,
        sionna_paths.tau.jax(),
    )

    chex.assert_trees_all_close(
        dm.aod_el,
        jnp.rad2deg(sionna_paths.theta_t.jax()),
    )

    chex.assert_trees_all_close(
        dm.aod_az,
        jnp.rad2deg(sionna_paths.phi_t.jax()),
        atol=3e-5,
    )

    chex.assert_trees_all_close(
        dm.aoa_el,
        jnp.rad2deg(sionna_paths.theta_r.jax()),
        atol=1e-4,
    )

    # We move the discontinuity to 360°->0° as angles are close to 180°
    chex.assert_trees_all_close(
        (dm.aoa_az + 360) % 360,
        (jnp.rad2deg(sionna_paths.phi_r.jax()) + 360) % 360,
        atol=1e-4,
    )

    a, tau = sionna_paths.cir(normalize_delays=False, out_type="jax")

    chex.assert_trees_all_close(
        dm.delay,
        tau,
    )

    a = a[:, 0, :, 0, :, :]  # Take only the first TX and RX polarization
    a = a[..., 0]  # Take only the first time instant

    # TODO: Understand why phase and power are not matching

    del a

    # chex.assert_trees_all_equal(
    #     dm.phase,
    #     jnp.angle(a, deg=True)
    # )

    # chex.assert_trees_all_equal(
    #     dm.power,
    #     10.0 * jnp.log10(jnp.abs(a)**2 / z_0),
    # )
