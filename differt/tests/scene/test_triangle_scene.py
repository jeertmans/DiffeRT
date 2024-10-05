import os
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from differt.geometry.utils import assemble_paths, normalize
from differt.scene.sionna import (
    get_sionna_scene,
    list_sionna_scenes,
)
from differt.scene.triangle_scene import TriangleScene
from differt_core.scene.sionna import SionnaScene


class TestTriangleScene:
    def test_load_xml(self, sionna_folder: Path) -> None:
        # Sionne scenes are all triangle scenes.
        for scene_name in list_sionna_scenes(folder=sionna_folder):
            file = get_sionna_scene(scene_name, folder=sionna_folder)
            scene = TriangleScene.load_xml(file)
            sionna_scene = SionnaScene.load_xml(file)

            assert scene.mesh.object_bounds is not None
            assert len(scene.mesh.object_bounds) == len(sionna_scene.shapes)

    @pytest.mark.parametrize(
        ("order", "expected_path_vertices", "expected_objects"),
        [
            (0, jnp.empty((1, 0, 3)), jnp.array([[0, 0]])),
            (
                1,
                jnp.array([
                    [[-0.06917738914489746, 14.946798324584961, 8.24851131439209]]
                ]),
                jnp.array([[0, 8, 0]]),
            ),
            (
                2,
                jnp.array([
                    [
                        [-0.125960111618042, 14.946202278137207, 13.787875175476074],
                        [-0.04232808202505112, 5.0, 5.629261016845703],
                    ]
                ]),
                jnp.array([[0, 9, 22, 0]]),
            ),
            (
                3,
                jnp.array([
                    [
                        [-0.17936798930168152, 14.945640563964844, 16.1051082611084],
                        [-0.14879928529262543, 5.0, 10.249288558959961],
                        [-0.11822860687971115, 14.946282386779785, 4.393090724945068],
                    ]
                ]),
                jnp.array([[0, 9, 22, 8, 0]]),
            ),
            (
                4,
                jnp.array([
                    [
                        [-0.233406662940979, 14.945074081420898, 17.426870346069336],
                        [-0.25651583075523376, 5.0, 12.884565353393555],
                        [-0.2796238660812378, 14.944588661193848, 8.342482566833496],
                        [-0.09397590905427933, 5.0, 3.799619674682617],
                    ]
                ]),
                jnp.array([[0, 9, 23, 8, 22, 0]]),
            ),
        ],
    )
    def test_compute_paths_on_advanced_path_tracing_example(
        self,
        order: int,
        expected_path_vertices: Array,
        expected_objects: Array,
        advanced_path_tracing_example_scene: TriangleScene,
    ) -> None:
        scene = advanced_path_tracing_example_scene
        expected_path_vertices = assemble_paths(
            scene.transmitters[None, :],
            expected_path_vertices,
            scene.receivers[None, :],
        )

        with jax.debug_nans(False):  # noqa: FBT003
            got = scene.compute_paths(order)

        chex.assert_trees_all_close(got.masked_vertices, expected_path_vertices)  # type: ignore[reportAttributeAccessIssue]
        chex.assert_trees_all_equal(got.masked_objects, expected_objects)  # type: ignore[reportAttributeAccessIssue]

        normals = jnp.take(scene.mesh.normals, got.masked_objects[..., 1:-1], axis=0)  # type: ignore[reportAttributeAccessIssue]

        rays = jnp.diff(got.masked_vertices, axis=-2)  # type: ignore[reportAttributeAccessIssue]

        rays = normalize(rays)[0]

        indicents = rays[..., :-1, :]
        reflecteds = rays[..., +1:, :]

        dot_incidents = jnp.sum(-indicents * normals, axis=-1)
        dot_reflecteds = jnp.sum(reflecteds * normals, axis=-1)

        chex.assert_trees_all_close(dot_incidents, dot_reflecteds)

    @pytest.mark.parametrize(("m_tx", "n_tx"), [(5, None), (3, 4)])
    @pytest.mark.parametrize(("m_rx", "n_rx"), [(2, None), (1, 6)])
    @pytest.mark.parametrize("pmap", [False, True, "cpu"])
    def test_compute_paths_on_grid(
        self,
        m_tx: int,
        n_tx: int | None,
        m_rx: int,
        n_rx: int | None,
        pmap: bool | str,
        advanced_path_tracing_example_scene: TriangleScene,
    ) -> None:
        scene = advanced_path_tracing_example_scene
        scene = scene.with_transmitters_grid(m_tx, n_tx)
        scene = scene.with_receivers_grid(m_rx, n_rx)
        paths = scene.compute_paths(order=1, pmap=pmap)

        if n_tx is None:
            n_tx = m_tx
        if n_rx is None:
            n_rx = m_rx

        num_path_candidates = scene.mesh.triangles.shape[0]

        chex.assert_shape(
            paths.vertices,  # type: ignore[reportAttributeAccessIssue]
            (n_tx, m_tx, n_rx, m_rx, num_path_candidates, 3, 3),
        )

    @pytest.mark.parametrize(
        ("m_tx", "n_tx", "m_rx", "n_rx", "expectation"),
        [
            (8, 8, 1, 1, does_not_raise()),
            (1, 1, 8, 8, does_not_raise()),
            (4, 2, 1, 1, does_not_raise()),
            (1, 1, 2, 4, does_not_raise()),
            pytest.param(
                7,
                1,
                1,
                1,
                pytest.raises(ValueError, match="Found 8 devices available"),
                marks=pytest.mark.xfail(
                    reason="Faking the number of devices does not seem to work."
                ),
            ),
            pytest.param(
                1,
                2,
                4,
                1,
                pytest.raises(ValueError, match="Found 8 devices available"),
                marks=pytest.mark.xfail(
                    reason="Faking the number of devices does not seem to work."
                ),
            ),
        ],
    )
    def test_compute_paths_pmap(
        self,
        m_tx: int,
        n_tx: int,
        m_rx: int,
        n_rx: int,
        expectation: AbstractContextManager[Exception],
        advanced_path_tracing_example_scene: TriangleScene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        scene = advanced_path_tracing_example_scene
        scene = scene.with_transmitters_grid(m_tx, n_tx)
        scene = scene.with_receivers_grid(m_rx, n_rx)

        with monkeypatch.context() as m, expectation:
            flags = os.environ.get("XLA_FLAGS", "")
            m.setenv("XLA_FLAGS", flags + " --xla_force_host_platform_device_count=8")

            paths = scene.compute_paths(order=1, pmap="cpu")

            num_path_candidates = scene.mesh.triangles.shape[0]

            chex.assert_shape(
                paths.vertices,  # type: ignore[reportAttributeAccessIssue]
                (n_tx, m_tx, n_rx, m_rx, num_path_candidates, 3, 3),
            )

    @pytest.mark.parametrize("backend", ["vispy", "matplotlib", "plotly"])
    def test_plot(
        self,
        backend: str,
        sionna_folder: Path,
    ) -> None:
        file = get_sionna_scene("simple_street_canyon", folder=sionna_folder)
        scene = TriangleScene.load_xml(file)

        tx = jnp.array([[0.0, 0.0, 0.0]])
        rx = jnp.array([[1.0, 1.0, 1.0]])

        scene = eqx.tree_at(lambda s: s.transmitters, scene, tx)
        scene = eqx.tree_at(lambda s: s.receivers, scene, rx)

        if backend == "matplotlib":
            # TODO: remove me when draw_markers is implemented
            scene.mesh.plot(backend=backend)
        else:
            scene.plot(backend=backend)
