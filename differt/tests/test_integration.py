from contextlib import nullcontext as does_not_raise

import chex
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from pytest_subtests import SubTests

from differt.em import materials
from differt.geometry import (
    TriangleMesh,
    assemble_paths,
    fibonacci_lattice,
    path_lengths,
)
from differt.rt import (
    first_triangles_hit_by_rays,
    rays_intersect_any_triangle,
    rays_intersect_triangles,
)
from differt.scene import TriangleScene


@pytest.mark.slow
def test_ray_casting() -> None:
    o3d = pytest.importorskip("open3d", reason="open3d not installed")

    knot_mesh = o3d.data.KnotMesh()
    o3d_mesh = o3d.io.read_triangle_mesh(knot_mesh.path).translate([50, 20, 10])

    o3d_mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
    o3d_mesh = o3d_mesh.compute_vertex_normals()  # This avoids a warning from Open3D
    o3d_mesh = o3d_mesh.compute_triangle_normals()

    mesh = TriangleMesh(
        vertices=o3d_mesh.vertex.positions.numpy(),
        triangles=o3d_mesh.triangle.indices.numpy(),
    )

    chex.assert_trees_all_close(
        mesh.bounding_box,
        jnp.stack(
            [
                o3d_mesh.get_min_bound().numpy(),
                o3d_mesh.get_max_bound().numpy(),
            ],
            axis=0,
        ),
    )

    chex.assert_trees_all_close(
        mesh.normals, o3d_mesh.triangle.normals.numpy(), atol=1e-6
    )

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d_mesh)

    ray_directions = fibonacci_lattice(1_000)
    ray_directions = fibonacci_lattice(50)
    ray_origins = jnp.zeros_like(ray_directions)

    o3d_rays = o3d.core.Tensor(
        np.concatenate((ray_origins, ray_directions), axis=-1),
        dtype=o3d.core.Dtype.Float32,
    )

    triangle_vertices = mesh.triangle_vertices

    triangles, t_hit = first_triangles_hit_by_rays(
        ray_origins, ray_directions, triangle_vertices
    )
    hit = triangles != -1
    triangles = triangles.astype(jnp.uint32)

    ans = scene.cast_rays(o3d_rays, nthreads=1)  # codespell:ignore ans

    chex.assert_trees_all_close(
        t_hit,
        ans["t_hit"].numpy(),  # codespell:ignore ans
        atol=1e-4,
    )
    chex.assert_trees_all_equal(
        jnp.where(hit, triangles, jnp.asarray(scene.INVALID_ID, dtype=jnp.uint32)),
        ans["primitive_ids"].numpy(),  # codespell:ignore ans
    )

    got_counts = rays_intersect_triangles(
        ray_origins[..., None, :], ray_directions[..., None, :], triangle_vertices
    )[1].sum(axis=-1)

    expected_counts = scene.count_intersections(o3d_rays, nthreads=1).numpy()

    chex.assert_trees_all_equal(
        got_counts,
        expected_counts,
    )

    scale = 100.0

    got_hit = rays_intersect_any_triangle(
        ray_origins,
        scale * ray_directions,
        triangle_vertices,
    )

    expected_hit = scene.test_occlusions(o3d_rays, tfar=scale, nthreads=1).numpy()

    chex.assert_trees_all_equal(
        got_hit,
        expected_hit,
    )


@pytest.mark.slow
def test_simple_street_canyon() -> None:
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
        pattern="tr38901",
        polarization="V",
    )

    sionna_scene.rx_array = sionna.rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
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

    max_order = 4

    sionna_solver = sionna.rt.PathSolver()
    sionna_paths = sionna_solver(sionna_scene, max_depth=max_order, refraction=False)
    sionna_path_objects = sionna_paths.objects.jax()
    sionna_path_vertices = sionna_paths.vertices.jax()

    max_depth = sionna_path_objects.shape[0]  # May differ from 'max_order'

    for order in range(max_depth + 1):
        paths = differt_scene.compute_paths(
            order=order,
            method="hybrid",
        )
        select = (sionna_path_objects == -1).sum(axis=0) == (max_depth - order)
        vertices = sionna_path_vertices[:order, select, :]
        vertices = jnp.moveaxis(vertices, 0, -2)
        vertices = assemble_paths(
            differt_scene.transmitters,
            vertices,
            differt_scene.receivers,
        )
        got_path_lengths = path_lengths(paths.masked_vertices)
        expected_path_lengths = path_lengths(vertices)
        # We check the sum of path lengths because Sionna orders the paths differently,
        # so we cannot compare them directly.
        chex.assert_trees_all_close(
            got_path_lengths.sum(),
            expected_path_lengths.sum(),
            atol=1e-5,
            custom_message=f"Mismatch for paths {order = }, differt = {paths.masked_vertices!r}, sionna = {vertices!r}.",
        )


def test_itu_materials(subtests: SubTests) -> None:
    mi = pytest.importorskip("mitsuba", reason="mitsuba not installed")
    try:
        mi.set_variant("llvm_ad_mono_polarized")
    except AttributeError:
        pytest.skip("Mitsuba variant 'llvm_ad_mono_polarized' not available")
    sionna = pytest.importorskip("sionna", reason="sionna not installed")

    for mat_name, differt_mat in materials.items():
        if not mat_name.startswith("itu_"):
            continue
        if mat_name == "itu_vacuum":
            continue  # Sionna removed it

        # We multiply by 1.1 to avoid checking on freq. limits, because Sionna will fail
        for f in 1.1 * np.logspace(9 - 2, 9 + 3, 21):
            differt_mat_relative_permittivity = differt_mat.relative_permittivity(f)
            differt_mat_conductivity = differt_mat.conductivity(f)

            if (differt_mat_relative_permittivity, differt_mat_conductivity) == (
                -1.0,
                -1.0,
            ):
                # Sionna now raises an error if any of the materials is not defined at the given frequency
                expectation = pytest.raises(
                    ValueError,
                    match=f"Properties of ITU material {mat_name.removeprefix('itu_')!r} are not defined for this frequency",
                )
            else:
                expectation = does_not_raise()

            with subtests.test(f"{mat_name} @ {f / 1e9} GHz"), expectation:
                sionna_mat_relative_permittivity, sionna_mat_conductivity = (
                    sionna.rt.radio_materials.itu.itu_material(
                        mat_name.removeprefix("itu_"), f
                    )
                )

                chex.assert_trees_all_close(
                    differt_mat.relative_permittivity(f),
                    sionna_mat_relative_permittivity,
                    custom_message=f"Mismatch for {mat_name = } @ {f / 1e9} GHz.",
                )

                chex.assert_trees_all_close(
                    differt_mat.conductivity(f),
                    sionna_mat_conductivity,
                    custom_message=f"Mismatch for {mat_name = } @ {f / 1e9} GHz.",
                )
