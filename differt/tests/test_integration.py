import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from differt.geometry import (
    TriangleMesh,
    assemble_paths,
    fibonacci_lattice,
)
from differt.rt import first_triangles_hit_by_rays
from differt.scene import TriangleScene


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

    ans = scene.cast_rays(o3d_rays)  # codespell:ignore ans

    chex.assert_trees_all_close(t_hit, ans["t_hit"].numpy(), atol=1e-4)  # codespell:ignore ans
    chex.assert_trees_all_equal(
        jnp.where(hit, triangles, jnp.asarray(scene.INVALID_ID, dtype=jnp.uint32)),
        ans["primitive_ids"].numpy(),  # codespell:ignore ans
    )


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
        with jax.debug_nans(False):  # noqa: FBT003
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
