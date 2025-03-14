import chex
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from differt.em import (
    Dipole,
    materials,
    pointing_vector,
    reflection_coefficients,
    sp_directions,
)
from differt.geometry import (
    TriangleMesh,
    assemble_paths,
    fibonacci_lattice,
    normalize,
)
from differt.rt import (
    first_triangles_hit_by_rays,
    rays_intersect_any_triangle,
    rays_intersect_triangles,
)
from differt.scene import TriangleScene
from differt.utils import dot


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


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:divide by zero encountered in log10:RuntimeWarning")
def test_coverage_maps() -> None:
    sionna = pytest.importorskip("sionna")
    file = sionna.rt.scene.simple_street_canyon

    sionna_scene = sionna.rt.load_scene(file)
    differt_scene = TriangleScene.load_xml(file)
    ant = Dipole(2.4e9)
    sionna_scene.frequency = ant.frequency

    sionna_scene.tx_array = sionna.rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="cross",
    )

    sionna_scene.rx_array = sionna.rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="cross",
    )

    tx = sionna.rt.Transmitter(
        name="tx", position=[-33.0, 0.0, 32.0], look_at=[20.0, 0.0, 32.0]
    )

    sionna_scene.add(tx)

    rx = sionna.rt.Receiver(name="rx", position=[20.0, 0.0, 1.5])

    sionna_scene.add(rx)
    tx.look_at(rx)

    # Compute coverage map
    max_depth = 0
    cm = sionna_scene.coverage_map(max_depth=max_depth, cm_cell_size=[5.0, 5.0])
    sionna_path_gain = cm.path_gain[0, :, :].numpy()
    db = 10 * jnp.log10(sionna_path_gain)
    vmin = -50
    vmax = -110
    vmin = vmax = None
    cm.show(vmin=vmin, vmax=vmax)
    import matplotlib.pyplot as plt

    plt.savefig("coverage_map.png")

    sionna_path_gain = cm.path_gain[0, :, :].numpy()

    transmitter = tx.position.numpy()
    receivers = cm.cell_centers.numpy()
    differt_scene = eqx.tree_at(
        lambda s: (s.transmitters, s.receivers),
        differt_scene,
        replace=(jnp.asarray(transmitter), jnp.asarray(receivers)),
    )

    print(f"{differt_scene.transmitters = }")

    E = jnp.zeros_like(receivers, dtype=jnp.complex64)
    B = jnp.zeros_like(E)

    print(f"{differt_scene = }")

    eta_r = jnp.array([
        materials[mat_name.removeprefix("mat-")].relative_permittivity(ant.frequency)
        for mat_name in differt_scene.mesh.material_names
    ])
    n_r = jnp.sqrt(eta_r)

    max_depth = 0

    for order in range(max_depth + 1):
        for paths in differt_scene.compute_paths(order=order, chunk_size=1_000):
            E_i, B_i = ant.fields(paths.vertices[..., 1, :])

            print(f"{E_i.shape = }")

            if order > 0:
                # [*batch num_path_candidates order]
                obj_indices = paths.objects[..., 1:-1]
                # [*batch num_path_candidates order]
                mat_indices = jnp.take(
                    differt_scene.mesh.face_materials, obj_indices, axis=0
                )
                # [*batch num_path_candidates order 3]
                obj_normals = jnp.take(differt_scene.mesh.normals, obj_indices, axis=0)
                # [*batch num_path_candidates order]
                obj_n_r = jnp.take(n_r, mat_indices, axis=0)
                # [*batch num_path_candidates order+1 3]
                path_segments = jnp.diff(paths.vertices, axis=-2)
                # [*batch num_path_candidates order+1 3],
                # [*batch num_path_candidates order+1 1]
                k, s = normalize(path_segments, keepdims=True)
                # [*batch num_path_candidates order 3]
                k_i = k[..., :-1, :]
                k_r = k[..., +1:, :]
                # [*batch num_path_candidates order 3]
                (e_i_s, e_i_p), (e_r_s, e_r_p) = sp_directions(k_i, k_r, obj_normals)
                # [*batch num_path_candidates order 1]
                cos_theta = dot(obj_normals, -k_i, keepdims=True)
                # [*batch num_path_candidates order 1]
                r_s, r_p = reflection_coefficients(obj_n_r[..., None], cos_theta)
                # [*batch num_path_candidates 1]
                r_s = jnp.prod(r_s, axis=-2)
                r_p = jnp.prod(r_p, axis=-2)
                # [*batch num_path_candidates order 3]
                (e_i_s, e_i_p), (e_r_s, e_r_p) = sp_directions(k_i, k_r, obj_normals)
                # [*batch num_path_candidates 1]
                E_i_s = dot(E_i, e_i_s[..., 0, :], keepdims=True)
                E_i_p = dot(E_i, e_i_p[..., 0, :], keepdims=True)
                B_i_s = dot(B_i, e_i_s[..., 0, :], keepdims=True)
                B_i_p = dot(B_i, e_i_p[..., 0, :], keepdims=True)
                # [*batch num_path_candidates 1]
                E_r_s = r_s * E_i_s
                E_r_p = r_p * E_i_p
                B_r_s = r_s * B_i_s
                B_r_p = r_p * B_i_p
                # [*batch num_path_candidates 3]
                E_r = E_r_s * e_r_s[..., -1, :] + E_r_p * e_r_p[..., -1, :]
                B_r = B_r_s * e_r_s[..., -1, :] + B_r_p * e_r_p[..., -1, :]
                # [*batch num_path_candidates 1]
                s_tot = s.sum(axis=-2)
                spreading_factor = s[..., 0, :] / s_tot  # Far-field approximation
                phase_shift = jnp.exp(1j * s_tot * ant.wavenumber)
                # [*batch num_path_candidates 3]
                E_r *= spreading_factor * phase_shift
                B_r *= spreading_factor * phase_shift
            else:
                # [*batch num_path_candidates 3]
                E_r = E_i
                B_r = B_i

            # [*batch 3]
            E += jnp.sum(E_r, axis=-2, where=paths.mask[..., None])
            B += jnp.sum(B_r, axis=-2, where=paths.mask[..., None])

    S = pointing_vector(E, B)
    P = ant.aperture * jnp.linalg.norm(S, axis=-1)
    differt_path_gain = P / ant.reference_power

    plt.figure()
    plt.imshow(10 * jnp.log10(differt_path_gain), origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Path gain [dB]")

    plt.xlabel("Cell index (X-axis)")
    plt.ylabel("Cell index (Y-axis)")
    plt.title("Path gain")
    plt.savefig("differt_path_gain.png")

    tol = 0.05
    chex.assert_trees_all_equal_comparator(
        lambda x, y: ((x == 0) ^ (y == 0)).sum() <= x.size * tol,
        lambda x,
        y,: f"Arrays {x} and {y} differ by more than {100 * tol:.2f}% of their elements.",
        sionna_path_gain,
        differt_path_gain,
    )

    chex.assert_trees_all_close(
        sionna_path_gain,
        differt_path_gain,
    )
