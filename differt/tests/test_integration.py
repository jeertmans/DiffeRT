import chex
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from differt.em import materials
from differt.geometry import (
    TriangleMesh,
    assemble_paths,
    fibonacci_lattice,
)
from differt.rt import (
    first_triangles_hit_by_rays,
    rays_intersect_any_triangle,
    rays_intersect_triangles,
)
from differt.scene import TriangleScene


@pytest.mark.slow
def test_ray_casting() -> None:
    o3d = pytest.importorskip("open3d")

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
    sionna = pytest.importorskip("sionna")
    file = sionna.rt.scene.simple_street_canyon

    sionna_scene = sionna.rt.load_scene(file)
    differt_scene = TriangleScene.load_xml(file)

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
        replace=jnp.asarray(tx.position.numpy()),
    )

    differt_scene = eqx.tree_at(
        lambda s: s.receivers,
        differt_scene,
        replace=jnp.asarray(rx.position.numpy()),
    )

    max_order = 4

    sionna_paths = sionna_scene.compute_paths(max_depth=max_order, method="exhaustive")
    sionna_path_objects = sionna_paths.objects.numpy()
    sionna_path_vertices = sionna_paths.vertices.numpy()

    max_depth = sionna_path_objects.shape[0]  # May differ from 'max_order'

    for order in range(max_depth + 1):
        paths = differt_scene.compute_paths(order=order)
        select = (sionna_path_objects == -1).sum(axis=0) == (max_depth - order)
        vertices = sionna_path_vertices[:order, select, :]
        vertices = np.moveaxis(vertices, 0, -2)
        vertices = assemble_paths(
            differt_scene.transmitters.reshape(1, 3),
            jnp.asarray(vertices),
            differt_scene.receivers.reshape(1, 3),
        )
        chex.assert_trees_all_close(
            paths.masked_vertices,
            vertices,
            atol=1e-5,
            custom_message=f"Mismatch for paths {order = }.",
        )


def test_itu_materials() -> None:
    sionna = pytest.importorskip("sionna")
    sionna_scene = sionna.rt.scene.Scene("__empty__")

    for mat_name, differt_mat in materials.items():
        if not mat_name.startswith("itu_"):
            continue

        if mat_name == "itu_vacuum":
            sionna_mat = sionna_scene.get("vacuum")
        else:
            sionna_mat = sionna_scene.get(mat_name)

        for f in np.logspace(9 - 2, 9 + 3, 21):
            sionna_scene.frequency = f

            chex.assert_trees_all_close(
                differt_mat.relative_permittivity(f),
                sionna_mat.relative_permittivity,
                custom_message=f"Mismatch for {mat_name = } @ {f / 1e9} GHz.",
            )

            chex.assert_trees_all_close(
                differt_mat.conductivity(f),
                sionna_mat.conductivity,
                custom_message=f"Mismatch for {mat_name = } @ {f / 1e9} GHz.",
            )
