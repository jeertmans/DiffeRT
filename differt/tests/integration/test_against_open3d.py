import chex
import jax.numpy as jnp
import numpy as np
import pytest

from differt.geometry import (
    Mesh,
    fibonacci_lattice,
    first_triangle_hit_by_ray,
    ray_intersect_any_triangle,
    ray_intersect_triangle,
)

o3d = pytest.importorskip("open3d", reason="open3d not installed")


@pytest.mark.slow
def test_ray_casting() -> None:
    knot_mesh = o3d.data.KnotMesh()
    o3d_mesh = o3d.t.io.read_triangle_mesh(knot_mesh.path).translate([50, 20, 10])
    o3d_mesh = o3d_mesh.compute_vertex_normals()  # This avoids a warning from Open3D
    o3d_mesh = o3d_mesh.compute_triangle_normals()

    mesh = Mesh(
        vertices=jnp.asarray(o3d_mesh.vertex.positions.numpy()),
        triangles=jnp.asarray(o3d_mesh.triangle.indices.numpy()),
    )

    chex.assert_trees_all_close(
        mesh.bounding_box,
        np.stack(
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

    ray_directions = fibonacci_lattice(50)
    ray_origins = jnp.zeros_like(ray_directions)

    o3d_rays = o3d.core.Tensor(
        np.concatenate((ray_origins, ray_directions), axis=-1),
        dtype=o3d.core.Dtype.Float32,
    )

    triangle_vertices = mesh.triangle_vertices

    triangles, t_hit = first_triangle_hit_by_ray(
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

    got_counts = ray_intersect_triangle(
        ray_origins[..., None, :], ray_directions[..., None, :], triangle_vertices
    )[1].sum(axis=-1)

    expected_counts = scene.count_intersections(o3d_rays, nthreads=1).numpy()

    chex.assert_trees_all_equal(
        got_counts,
        expected_counts,
    )

    scale = 100.0

    got_hit = ray_intersect_any_triangle(
        ray_origins,
        scale * ray_directions,
        triangle_vertices,
    )

    expected_hit = scene.test_occlusions(o3d_rays, tfar=scale, nthreads=1).numpy()

    chex.assert_trees_all_equal(
        got_hit,
        expected_hit,
    )
