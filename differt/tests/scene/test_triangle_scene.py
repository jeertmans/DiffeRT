from collections.abc import Iterator
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Literal

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Int, PRNGKeyArray
from pytest_subtests import SubTests

from differt.geometry import (
    Paths,
    TriangleMesh,
    assemble_paths,
    normalize,
    rotation_matrix_along_x_axis,
)
from differt.scene import (
    get_sionna_scene,
    list_sionna_scenes,
)
from differt.scene._triangle_scene import TriangleScene
from differt_core.scene import SionnaScene

from ..plotting.params import matplotlib, plotly, vispy


class TestTriangleScene:
    def test_load_xml(self, sionna_folder: Path, subtests: SubTests) -> None:
        # Sionne scenes are all triangle scenes
        for scene_name in list_sionna_scenes(folder=sionna_folder):
            with subtests.test(scene_name=scene_name):
                file = get_sionna_scene(scene_name, folder=sionna_folder)
                scene = TriangleScene.load_xml(file)
                sionna_scene = SionnaScene.load_xml(file)

                assert scene.mesh.object_bounds is not None
                assert len(scene.mesh.object_bounds) == len(sionna_scene.shapes)

    def test_from_sionna(self, sionna_folder: Path, subtests: SubTests) -> None:
        sionna = pytest.importorskip("sionna", reason="sionna not installed")
        for scene_name in list_sionna_scenes(folder=sionna_folder):
            with subtests.test(scene_name=scene_name):
                file = get_sionna_scene(scene_name, folder=sionna_folder)
                sionna_scene = sionna.rt.load_scene(file)
                differt_scene = TriangleScene.from_sionna(sionna_scene)
                assert differt_scene.mesh.num_triangles > 0
                assert differt_scene.mesh.face_colors is not None
                assert differt_scene.mesh.face_materials is not None

    def test_rotate(
        self, advanced_path_tracing_example_scene: TriangleScene, key: PRNGKeyArray
    ) -> None:
        angle = jax.random.uniform(key, (), minval=0, maxval=2 * jnp.pi)

        got = advanced_path_tracing_example_scene.rotate(
            rotation_matrix_along_x_axis(angle)
        ).rotate(rotation_matrix_along_x_axis(-angle))
        chex.assert_trees_all_close(got, advanced_path_tracing_example_scene, atol=1e-5)

        got = advanced_path_tracing_example_scene.rotate(
            rotation_matrix_along_x_axis(angle)
        ).rotate(rotation_matrix_along_x_axis(2 * jnp.pi - angle))
        chex.assert_trees_all_close(got, advanced_path_tracing_example_scene, atol=1e-4)

        got = advanced_path_tracing_example_scene.rotate(
            rotation_matrix_along_x_axis(0.0)
        )
        chex.assert_trees_all_close(got, advanced_path_tracing_example_scene)

    def test_scale(
        self, advanced_path_tracing_example_scene: TriangleScene, key: PRNGKeyArray
    ) -> None:
        scale_factor = jax.random.uniform(key, (), minval=1.5, maxval=2)

        got = advanced_path_tracing_example_scene.scale(scale_factor).scale(
            1 / scale_factor
        )
        chex.assert_trees_all_close(got, advanced_path_tracing_example_scene)

        got = advanced_path_tracing_example_scene.scale(1.0)
        chex.assert_trees_all_close(got, advanced_path_tracing_example_scene)

    def test_translate(
        self, advanced_path_tracing_example_scene: TriangleScene, key: PRNGKeyArray
    ) -> None:
        translation = jax.random.normal(key, (3,))

        got = advanced_path_tracing_example_scene.translate(translation).translate(
            -translation
        )
        chex.assert_trees_all_close(got, advanced_path_tracing_example_scene)

        got = advanced_path_tracing_example_scene.translate(jnp.zeros_like(translation))
        chex.assert_trees_all_close(got, advanced_path_tracing_example_scene)

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
    @pytest.mark.parametrize("assume_quads", [False, True])
    @pytest.mark.parametrize("mesh_mask", [False, True])
    @pytest.mark.parametrize(
        "method",
        ["exhaustive", "sbr", "hybrid"],
    )
    def test_compute_paths_on_advanced_path_tracing_example(
        self,
        order: int,
        expected_path_vertices: Array,
        expected_objects: Array,
        assume_quads: bool,
        mesh_mask: bool,
        method: Literal["exhaustive", "sbr", "hybrid"],
        advanced_path_tracing_example_scene: TriangleScene,
    ) -> None:
        scene = advanced_path_tracing_example_scene.set_assume_quads(assume_quads)
        expected_path_vertices = assemble_paths(
            scene.transmitters,
            expected_path_vertices,
            scene.receivers,
        )

        if assume_quads:
            expected_objects -= expected_objects % 2

        if mesh_mask:
            scene = eqx.tree_at(
                lambda s: s.mesh.mask,
                scene,
                jnp.ones(scene.mesh.triangles.shape[0], dtype=bool),
                is_leaf=lambda x: x is None,
            )

        got = scene.compute_paths(order, method=method, max_dist=1e-1)

        if method == "sbr":
            masked_vertices = got.masked_vertices
            masked_objects = got.masked_objects
            masked_objects -= masked_objects % 2
            expected_objects -= expected_objects % 2
            unique_objects = jnp.unique(masked_objects, axis=0)
            vertices = jnp.empty_like(
                masked_vertices,
                shape=(unique_objects.shape[0], *masked_vertices.shape[1:]),
            )
            for i, path_candidate in enumerate(unique_objects):
                vertices = vertices.at[i, ...].set(
                    masked_vertices.mean(
                        axis=0,
                        where=(masked_objects == path_candidate).all(axis=-1)[
                            ..., None, None
                        ],
                    )
                )

            got = Paths(vertices=vertices, objects=unique_objects)
            rtol = 0.52  # TODO: see if we can improve acc.
        else:
            rtol = 1e-6

        chex.assert_trees_all_close(
            got.masked_vertices, expected_path_vertices, rtol=rtol
        )
        chex.assert_trees_all_equal(got.masked_objects, expected_objects)

        normals = jnp.take(scene.mesh.normals, got.masked_objects[..., 1:-1], axis=0)

        rays = jnp.diff(got.masked_vertices, axis=-2)

        rays = normalize(rays)[0]

        indicents = rays[..., :-1, :]
        reflecteds = rays[..., +1:, :]

        dot_incidents = jnp.sum(-indicents * normals, axis=-1)
        dot_reflecteds = jnp.sum(reflecteds * normals, axis=-1)

        chex.assert_trees_all_close(dot_incidents, dot_reflecteds, rtol=rtol)

    @pytest.mark.parametrize(
        ("order", "expected_path_vertices", "expected_objects"),
        [
            (0, jnp.empty((1, 0, 3)), jnp.array([[0, 0]])),
            (
                1,
                jnp.array([
                    [[0.0, -8.613334655761719, 32.0]],
                    [[0.0, 9.571563720703125, 32.0]],
                    [[1.9073486328125e-06, 0.0, -0.030788421630859375]],
                ]),
                jnp.array([[0, 18, 0], [0, 38, 0], [0, 72, 0]]),
            ),
            (
                2,
                jnp.array([
                    [
                        [-11.579630851745605, -8.613335609436035, 32.0],
                        [10.420369148254395, 9.571564674377441, 32.0],
                    ],
                    [
                        [-10.420370101928711, 9.571562767028809, 32.0],
                        [11.579629898071289, -8.613335609436035, 32.0],
                    ],
                ]),
                jnp.array([[0, 19, 39, 0], [0, 38, 18, 0]]),
            ),
        ],
    )
    @pytest.mark.parametrize("assume_quads", [False, True])
    def test_compute_paths_on_simple_street_canyon(
        self,
        order: int,
        expected_path_vertices: Array,
        expected_objects: Array,
        assume_quads: bool,
        simple_street_canyon_scene: TriangleScene,
    ) -> None:
        scene = simple_street_canyon_scene.set_assume_quads(assume_quads)
        expected_path_vertices = assemble_paths(
            scene.transmitters,
            expected_path_vertices,
            scene.receivers,
        )

        if assume_quads:
            expected_objects -= expected_objects % 2

        got = scene.compute_paths(order)

        chex.assert_trees_all_close(
            got.masked_vertices, expected_path_vertices, atol=1e-5
        )
        chex.assert_trees_all_equal(got.masked_objects, expected_objects)

        normals = jnp.take(scene.mesh.normals, got.masked_objects[..., 1:-1], axis=0)

        rays = jnp.diff(got.masked_vertices, axis=-2)

        rays = normalize(rays)[0]

        indicents = rays[..., :-1, :]
        reflecteds = rays[..., +1:, :]

        dot_incidents = jnp.sum(-indicents * normals, axis=-1)
        dot_reflecteds = jnp.sum(reflecteds * normals, axis=-1)

        chex.assert_trees_all_close(dot_incidents, dot_reflecteds)

    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    @pytest.mark.parametrize("assume_quads", [False, True])
    def test_compute_paths_with_path_candidates_matches_exhaustive(self, order: int, assume_quads: bool, simple_street_canyon_scene: TriangleScene) -> None:
        scene = simple_street_canyon_scene.set_assume_quads(assume_quads)
        expected = scene.compute_paths(order=order)
        path_candidates = expected.objects[:, 1:-1]
        got = scene.compute_paths(path_candidates=path_candidates)
        chex.assert_trees_all_equal(got.masked_vertices, expected.masked_vertices)

        # Also check if passing a single path candidate works
        path_candidates = expected.masked().objects[:, 1:-1]

        for path_candidate in path_candidates:
            got = scene.compute_paths(
                path_candidates=path_candidate[None, :],
            )
            assert got.mask is not None and got.mask.size == 1
            chex.assert_trees_all_equal(got.mask.squeeze(), True, custom_message=f"Path candidate should be valid: {path_candidate}")

    @pytest.mark.xfail(reason="Not yet (correctly) implemented.")
    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    @pytest.mark.parametrize(
        "method",
        [
            "exhaustive",
            "sbr",
            "hybrid",
        ],
    )
    @pytest.mark.parametrize("chunk_size", [None, 1000])
    @pytest.mark.parametrize("assume_quads", [False, True])
    @pytest.mark.parametrize("mesh_mask", [False, True])
    def test_compute_paths_with_smoothing(
        self,
        order: int | None,
        method: Literal["exhaustive", "sbr", "hybrid"],
        chunk_size: int | None,
        assume_quads: bool,
        mesh_mask: bool,
        simple_street_canyon_scene: TriangleScene,
    ) -> None:
        # TODO: fixme
        scene = simple_street_canyon_scene.set_assume_quads(assume_quads)

        if mesh_mask:
            scene = eqx.tree_at(
                lambda s: s.mesh.mask,
                scene,
                jnp.ones(scene.mesh.triangles.shape[0], dtype=bool),
                is_leaf=lambda x: x is None,
            )

        expected = scene.compute_paths(  # ty: ignore[no-matching-overload]
            order=order,
            chunk_size=chunk_size,
            method=method,
        )

        if method != "exhaustive":
            expectation = pytest.warns(
                UserWarning,
                match="Argument 'smoothing' is currently ignored when 'method' is not set to 'exhaustive'",
            )
        else:
            expectation = does_not_raise()

        with expectation:
            got = scene.compute_paths(  # ty: ignore[no-matching-overload]
                order=order,
                method=method,
                chunk_size=chunk_size,
                smoothing_factor=1000.0,
            )

        assert type(got) is type(expected)

        if isinstance(got, Iterator):
            for got_paths, expected_paths in zip(got, expected, strict=True):
                chex.assert_trees_all_close(got_paths, expected_paths)
        else:
            chex.assert_trees_all_close(got, expected)

    @pytest.mark.parametrize(
        ("order", "chunk_size", "path_candidates", "method", "expectation"),
        [
            (0, None, None, "exhaustive", does_not_raise()),
            (0, 1000, None, "exhaustive", does_not_raise()),
            (
                None,
                None,
                jnp.empty((1, 0), dtype=jnp.int32),
                "exhaustive",
                does_not_raise(),
            ),
            (
                0,
                None,
                jnp.empty((1, 0), dtype=jnp.int32),
                "exhaustive",
                pytest.raises(ValueError, match="You must specify one of"),
            ),
            (
                None,
                1000,
                jnp.empty((1, 0), dtype=jnp.int32),
                "exhaustive",
                pytest.warns(UserWarning, match="Argument 'chunk_size' is ignored"),
            ),
            (
                None,
                None,
                jnp.empty((1, 0), dtype=jnp.int32),
                "sbr",
                pytest.raises(ValueError, match="Argument 'order' is required"),
            ),
            (
                None,
                None,
                jnp.empty((1, 0), dtype=jnp.int32),
                "hybrid",
                pytest.raises(ValueError, match="Argument 'order' is required"),
            ),
        ],
    )
    @pytest.mark.parametrize("assume_quads", [False, True])
    def test_compute_paths_on_empty_scene(
        self,
        order: int | None,
        chunk_size: int | None,
        path_candidates: Int[Array, "num_path_candidates order"] | None,
        method: Literal["exhaustive", "sbr", "hybrid"],
        expectation: AbstractContextManager[Exception],
        assume_quads: bool,
        key: PRNGKeyArray,
    ) -> None:
        # TODO: tests and fix issue in higher-order
        key_tx, key_rx = jax.random.split(key, 2)

        transmitters = jax.random.uniform(key_tx, (1, 3))

        receivers = jax.random.uniform(key_rx, (1, 3))

        scene = TriangleScene(
            transmitters=transmitters, receivers=receivers
        ).set_assume_quads(assume_quads)
        expected_path_vertices = assemble_paths(
            transmitters[:, None, None, :],
            receivers[None, :, None, :],
        )

        with expectation:
            got = scene.compute_paths(  # ty: ignore[no-matching-overload]
                order=order,
                chunk_size=chunk_size,
                path_candidates=path_candidates,
                method=method,
            )

            paths = next(got) if isinstance(got, Iterator) else got

            chex.assert_trees_all_close(paths.vertices, expected_path_vertices)

    @pytest.mark.parametrize(("m_tx", "n_tx"), [(5, None), (3, 4)])
    @pytest.mark.parametrize(("m_rx", "n_rx"), [(2, None), (1, 6)])
    def test_compute_paths_on_grid(
        self,
        m_tx: int,
        n_tx: int | None,
        m_rx: int,
        n_rx: int | None,
        advanced_path_tracing_example_scene: TriangleScene,
    ) -> None:
        scene = advanced_path_tracing_example_scene
        scene = scene.with_transmitters_grid(m_tx, n_tx)
        scene = scene.with_receivers_grid(m_rx, n_rx)

        paths = scene.compute_paths(order=1)

        if n_tx is None:
            n_tx = m_tx
        if n_rx is None:
            n_rx = m_rx

        num_path_candidates = scene.mesh.triangles.shape[0]

        chex.assert_shape(
            paths.vertices,
            (n_tx, m_tx, n_rx, m_rx, num_path_candidates, 3, 3),
        )

    @pytest.mark.parametrize("backend", [vispy, matplotlib, plotly])
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

    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    @pytest.mark.parametrize(
        "method",
        [
            "exhaustive",
            "sbr",
            "hybrid",
        ],
    )
    def test_compute_paths_with_mesh_mask_matches_sub_mesh_without_mask(
        self,
        order: int,
        method: Literal["exhaustive", "sbr", "hybrid"],
    ) -> None:
        # This test checks that the mask is correctly applied to the mesh
        # and that the path obtained with the mask is the same as the one obtained
        # on the 'subset' of the mesh.
        outer_mesh = TriangleMesh.box(10.0, 10.0, 10.0)
        inner_mesh = TriangleMesh.box(4.0, 4.0, 4.0)
        mesh = outer_mesh + inner_mesh
        mask = jnp.concatenate((
            jnp.ones((outer_mesh.num_triangles,), dtype=bool),  # Select outer mesh
            jnp.zeros(
                (inner_mesh.num_triangles,), dtype=bool
            ),  # Do not select inner mesh
        ))
        mesh = eqx.tree_at(lambda m: m.mask, mesh, mask, is_leaf=lambda x: x is None)
        tx = jnp.array([[-5.0, 0.0, 0.0]])
        rx = jnp.array([[+5.0, 0.0, 0.0]])
        got = (
            TriangleScene(
                transmitters=tx,
                receivers=rx,
                mesh=mesh,
            )
            .compute_paths(order, method=method)
            .masked()
        )
        expected = (
            TriangleScene(
                transmitters=tx,
                receivers=rx,
                mesh=outer_mesh,
            )
            .compute_paths(order, method=method)
            .masked()
        )

        chex.assert_trees_all_equal(got, expected)

    @pytest.mark.parametrize("method", ["exhaustive", "hybrid"])
    @pytest.mark.parametrize("assume_quads", [False, True])
    def test_compute_paths_with_no_path_candidate(
        self,
        method: Literal["exhaustive", "hybrid"],
        assume_quads: bool,
    ) -> None:
        mesh = TriangleMesh.box(6.0, 6.0, 6.0)

        mask = jnp.zeros((mesh.num_triangles,), dtype=bool)
        mesh = eqx.tree_at(lambda m: m.mask, mesh, mask, is_leaf=lambda x: x is None)

        tx = jnp.array([-1.5, 0.0, 0.0])
        rx = jnp.array([+1.5, 0.0, 0.0])

        scene = TriangleScene(
            transmitters=tx,
            receivers=rx,
            mesh=mesh,
        ).set_assume_quads(assume_quads)

        paths = scene.compute_paths(
            order=1,
            method=method,
            disconnect_inactive_triangles=True,
        )

        assert paths.vertices.shape[0] == 0, (
            "No active triangles should lead to no path candidates."
        )

    @pytest.mark.parametrize("assume_quads", [False, True])
    def test_compute_paths_with_disconnect_inactive_triangles(
        self,
        assume_quads: bool,
    ) -> None:
        # Create a scene with a masked mesh
        outer_mesh = TriangleMesh.box(6.0, 6.0, 6.0)
        inner_mesh = TriangleMesh.box(1.0, 1.0, 1.0)
        mesh = outer_mesh + inner_mesh

        # Mask to keep only the outer mesh (disconnect inner mesh)
        mask = jnp.concatenate((
            jnp.ones((outer_mesh.num_triangles,), dtype=bool),
            jnp.zeros((inner_mesh.num_triangles,), dtype=bool),
        ))
        mesh = eqx.tree_at(lambda m: m.mask, mesh, mask, is_leaf=lambda x: x is None)

        tx = jnp.array([-1.5, 0.0, 0.0])
        rx = jnp.array([+1.5, 0.0, 0.0])

        scene = TriangleScene(
            transmitters=tx,
            receivers=rx,
            mesh=mesh,
        ).set_assume_quads(assume_quads)

        # Compute paths with and without disconnecting inactive triangles
        paths = scene.compute_paths(
            order=1,
            disconnect_inactive_triangles=False,
        )
        paths_disc = scene.compute_paths(
            order=1,
            disconnect_inactive_triangles=True,
        )

        assert paths.vertices.shape[0] > paths_disc.vertices.shape[0], (
            "Disconnecting inactive triangles should reduce the number of path candidates."
        )

    @pytest.mark.parametrize("assume_quads", [False, True])
    def test_compute_paths_hybrid_always_disconnects(
        self,
        assume_quads: bool,
    ) -> None:
        # For hybrid method, inactive triangles are always disconnected regardless
        # of the disconnect_inactive_triangles parameter
        outer_mesh = TriangleMesh.box(6.0, 6.0, 6.0)
        inner_mesh = TriangleMesh.box(1.0, 1.0, 1.0)
        mesh = outer_mesh + inner_mesh

        # Mask to keep only the outer mesh
        mask = jnp.concatenate((
            jnp.ones((outer_mesh.num_triangles,), dtype=bool),
            jnp.zeros((inner_mesh.num_triangles,), dtype=bool),
        ))
        mesh = eqx.tree_at(lambda m: m.mask, mesh, mask, is_leaf=lambda x: x is None)

        tx = jnp.array([-1.5, 0.0, 0.0])
        rx = jnp.array([+1.5, 0.0, 0.0])

        scene = TriangleScene(
            transmitters=tx,
            receivers=rx,
            mesh=mesh,
        ).set_assume_quads(assume_quads)

        # Test with both True and False - should get same results for hybrid method
        paths_true = scene.compute_paths(
            order=1, method="hybrid", disconnect_inactive_triangles=True
        )

        paths_false = scene.compute_paths(
            order=1, method="hybrid", disconnect_inactive_triangles=False
        )

        # Should get the same results since hybrid always disconnects inactive triangles
        chex.assert_trees_all_equal(paths_true, paths_false)
