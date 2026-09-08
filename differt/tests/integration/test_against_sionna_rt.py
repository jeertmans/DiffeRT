from pathlib import Path

import chex
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from pytest_subtests import SubTests

from differt.em import (
    InteractionType,
    MaterialsDict,
    compute_received_fields,
    compute_received_power,
    materials,
)
from differt.geometry import (
    Scene,
    assemble_path,
    path_length,
)
from differt_core.geometry import SionnaScene

mi = pytest.importorskip("mitsuba", reason="mitsuba not installed")
try:
    mi.set_variant("llvm_ad_mono_polarized")
except (AttributeError, ImportError, RuntimeError):
    pytest.skip(
        "Mitsuba variant 'llvm_ad_mono_polarized' not available",
        allow_module_level=True,
    )
sionna = pytest.importorskip("sionna", reason="sionna not installed")


@pytest.mark.slow
def test_simple_street_canyon() -> None:
    file = sionna.rt.scene.simple_street_canyon

    sionna_scene = sionna.rt.load_scene(file)
    differt_scene = Scene.load_xml(
        file, materials=MaterialsDict(materials)
    ).set_assume_quads()  # Faster RT

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
        paths = differt_scene.trace_paths(
            order=order,
            solver="hybrid",
        )
        select = (sionna_path_objects == -1).sum(axis=0) == (max_depth - order)
        vertices = sionna_path_vertices[:order, select, :]
        vertices = jnp.moveaxis(vertices, 0, -2)
        vertices = assemble_path(
            differt_scene.transmitters,
            vertices,
            differt_scene.receivers,
        )
        got_path_lengths = path_length(paths.masked_vertices)
        expected_path_lengths = path_length(vertices)
        # We check the sum of path lengths because Sionna orders the paths differently,
        # so we cannot compare them directly.
        chex.assert_trees_all_close(
            got_path_lengths.sum(),
            expected_path_lengths.sum(),
            atol=1e-5,
            custom_message=f"Mismatch for paths {order = }, differt = {paths.masked_vertices!r}, sionna = {vertices!r}.",
        )


@pytest.mark.slow
def test_simple_street_canyon_pure_diffraction() -> None:
    """Test order 1 diffraction paths against Sionna RT ground truth."""
    file = sionna.rt.scene.simple_street_canyon

    sionna_scene = sionna.rt.load_scene(file)
    differt_scene = Scene.load_xml(file, materials=MaterialsDict(materials))

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
    rx = sionna.rt.Receiver(name="rx", position=[20.0, 0.0, 2.0], orientation=[0, 0, 0])
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

    solver = sionna.rt.PathSolver()
    s_paths = solver(
        sionna_scene,
        max_depth=1,
        los=False,
        specular_reflection=False,
        refraction=False,
        diffraction=True,
        edge_diffraction=True,
    )
    s_verts = s_paths.vertices.jax()[0, 0, 0, :, :]

    d_paths = differt_scene.trace_paths(
        order=1,
        allowed_interactions=frozenset({InteractionType.DIFFRACTION}),
        solver="hybrid",
    )
    d_verts = d_paths.vertices[d_paths.mask, 1, :]
    unique_d_verts = jnp.unique(jnp.round(d_verts, 3), axis=0)

    # 10 rooftop/wall edges match Sionna
    dist_matrix = jnp.linalg.norm(
        s_verts[:, None, :] - unique_d_verts[None, :, :], axis=-1
    )
    min_dists = dist_matrix.min(axis=-1)
    matching_rooftop = min_dists < 1e-2
    assert int(matching_rooftop.sum()) == 10
    chex.assert_trees_all_close(min_dists[matching_rooftop].max(), 0.0, atol=1e-3)


@pytest.mark.slow
def test_simple_street_canyon_mixed_diffraction_reflection() -> None:
    """Test order 2 mixed diffraction-reflection paths against Sionna RT ground truth."""
    file = sionna.rt.scene.simple_street_canyon

    sionna_scene = sionna.rt.load_scene(file)
    differt_scene = Scene.load_xml(file, materials=MaterialsDict(materials))

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
    rx = sionna.rt.Receiver(name="rx", position=[20.0, 0.0, 2.0], orientation=[0, 0, 0])
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

    solver = sionna.rt.PathSolver()
    s_paths = solver(
        sionna_scene,
        max_depth=2,
        los=False,
        specular_reflection=True,
        refraction=False,
        diffraction=True,
        edge_diffraction=True,
    )
    inter = s_paths.interactions.jax()[:, 0, 0, :]
    has_diff = (inter == 8).any(axis=0)
    has_refl = (inter == 1).any(axis=0)
    is_order2 = inter[1] > 0
    mixed_order2 = has_diff & has_refl & is_order2

    s_mixed_verts = jnp.moveaxis(
        s_paths.vertices.jax()[:2, 0, 0, mixed_order2, :], 0, 1
    )

    d_paths = differt_scene.trace_paths(
        order=2,
        allowed_interactions=frozenset({
            InteractionType.REFLECTION,
            InteractionType.DIFFRACTION,
        }),
        max_diffractions=1,
        solver="hybrid",
    )
    d_itypes = d_paths.interaction_types[d_paths.mask]
    d_has_diff = (d_itypes == InteractionType.DIFFRACTION).any(axis=-1)
    d_has_refl = (d_itypes == InteractionType.REFLECTION).any(axis=-1)
    d_mixed_verts = d_paths.vertices[d_paths.mask][d_has_diff & d_has_refl, 1:-1, :]

    path_diffs = jnp.linalg.norm(
        s_mixed_verts[:, None, :, :] - d_mixed_verts[None, :, :, :], axis=-1
    ).max(axis=-1)
    min_path_dists = path_diffs.min(axis=-1)
    matching = min_path_dists < 0.05
    assert int(matching.sum()) == len(s_mixed_verts) == 23
    chex.assert_trees_all_close(min_path_dists.max(), 0.0, atol=2e-3)


def _differt_color(itu_type: str, tmp_path: Path) -> tuple[float, float, float]:
    """Parse the color DiffeRT assigns to an ITU radio material, via a minimal Sionna scene."""
    xml = f"""<scene version="2.1.0">
  <bsdf type="itu-radio-material" id="mat-0">
    <string name="type" value="{itu_type}"/>
  </bsdf>
  <shape type="ply" id="mesh-0">
    <string name="filename" value="dummy.ply"/>
    <ref id="mat-0"/>
  </shape>
</scene>
"""
    file = tmp_path / "scene.xml"
    file.write_text(xml)
    return tuple(SionnaScene.load_xml(file).materials["mat-0"].color)


def test_itu_materials(subtests: SubTests, tmp_path: Path) -> None:
    for differt_mat in materials.values():
        # `materials` maps both official ITU names (e.g., "Wood") and Sionna-style
        # aliases (e.g., "itu_wood") to the same `Material` instance, but iterating
        # only sees the primary (official) name, so we look up the alias instead.
        mat_name = next((a for a in differt_mat.aliases if a.startswith("itu_")), None)
        if mat_name is None:
            continue

        itu_type = mat_name.removeprefix("itu_")

        if mat_name != "itu_vacuum":
            with subtests.test(f"{mat_name} color"):
                got_color = _differt_color(itu_type, tmp_path)
                expected_color = sionna.rt.ITURadioMaterial.ITU_MATERIAL_COLORS[
                    itu_type
                ]
                chex.assert_trees_all_close(got_color, expected_color, atol=1e-6)

        # We multiply by 1.1 to avoid checking on freq. limits, because Sionna will fail
        for f in 1.1 * np.logspace(9 - 2, 9 + 3, 21):
            differt_mat_relative_permittivity = differt_mat.relative_permittivity(f)
            differt_mat_conductivity = differt_mat.conductivity(f)
            differt_out_of_range = (
                differt_mat_relative_permittivity,
                differt_mat_conductivity,
            ) == (-1.0, -1.0)

            with subtests.test(f"{mat_name} @ {f / 1e9} GHz"):
                try:
                    sionna_mat_relative_permittivity, sionna_mat_conductivity = (
                        sionna.rt.radio_materials.itu.itu_material(itu_type, f)
                    )
                except ValueError as exc:
                    if differt_out_of_range:
                        # Both sides agree this frequency is out of range.
                        continue
                    # DiffeRT treats 'vacuum' as valid at any frequency (it has no
                    # measurement-derived validity range like other materials),
                    # while Sionna RT restricts it to ITU-R P.2040-4's [0.001, 100] GHz.
                    if mat_name == "itu_vacuum" and not (1e6 <= f <= 1e11):
                        continue

                    pytest.fail(
                        f"Sionna RT doesn't cover {f / 1e9:.3g} GHz for {itu_type!r} "
                        f"({exc}), but DiffeRT has a value for it."
                    )
                else:
                    if differt_out_of_range:
                        pytest.fail(
                            f"DiffeRT considers {itu_type!r} undefined at "
                            f"{f / 1e9:.3g} GHz, but Sionna RT returned a value; "
                            "DiffeRT's frequency-range logic may need updating."
                        )

                    chex.assert_trees_all_close(
                        differt_mat_relative_permittivity,
                        sionna_mat_relative_permittivity,
                        custom_message=f"Mismatch for {mat_name = } @ {f / 1e9} GHz.",
                    )
                    chex.assert_trees_all_close(
                        differt_mat_conductivity,
                        sionna_mat_conductivity,
                        custom_message=f"Mismatch for {mat_name = } @ {f / 1e9} GHz.",
                    )


@pytest.mark.slow
def test_received_power_matches_sionna() -> None:
    # Load simple street canyon scene
    file = sionna.rt.scene.simple_street_canyon
    sionna_scene = sionna.rt.load_scene(file)
    differt_scene = Scene.load_xml(
        file, materials=MaterialsDict(materials)
    ).set_assume_quads()

    # Configure transmitter and receiver antenna array
    # We use isotropic pattern with vertical polarization (V)
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

    # Position Transmitter and Receiver
    tx_pos = [-33.0, 0.0, 32.0]
    rx_pos = [20.0, 0.0, 2.0]
    tx = sionna.rt.Transmitter(name="tx", position=tx_pos)
    rx = sionna.rt.Receiver(name="rx", position=rx_pos)
    sionna_scene.add(tx)
    sionna_scene.add(rx)

    # Solve paths using Sionna
    # Limit to specular reflections only and match max depth
    max_depth = 2
    sionna_solver = sionna.rt.PathSolver()
    sionna_paths = sionna_solver(sionna_scene, max_depth=max_depth)
    a_cir, _ = sionna_paths.cir(normalize_delays=False, out_type="numpy")
    a_sionna = jnp.asarray(a_cir[0, 0, 0, 0, :, 0])

    # Calculate received power using Sionna (coherent and non-coherent)
    power_coherent_sionna = 10.0 * jnp.log10(jnp.abs(jnp.sum(a_sionna)) ** 2)
    power_non_coherent_sionna = 10.0 * jnp.log10(jnp.sum(jnp.abs(a_sionna) ** 2))

    # Setup matching scenario in DiffeRT
    differt_scene = eqx.tree_at(
        lambda s: s.transmitters,
        differt_scene,
        replace=jnp.asarray([tx_pos]),
    )
    differt_scene = eqx.tree_at(
        lambda s: s.receivers,
        differt_scene,
        replace=jnp.asarray([rx_pos]),
    )

    # Compute paths in DiffeRT
    fields_list = []
    for order in range(max_depth + 1):
        paths = differt_scene.trace_paths(order=order)
        f = compute_received_fields(
            paths,
            differt_scene.mesh,
            frequency=3.5e9,
            tx_polarization="V",
            rx_polarization="V",
        )
        fields_list.append(f.reshape(-1))

    all_fields = jnp.concatenate(fields_list)
    # Remove invalid paths (zero fields)
    all_fields = all_fields[jnp.abs(all_fields) > 1e-12]

    # Calculate received power in DiffeRT with z_0=1.0 to match normalization
    power_coherent_differt = compute_received_power(
        all_fields, coherent=True, axis=0, z_0=1.0
    )
    power_non_coherent_differt = compute_received_power(
        all_fields, coherent=False, axis=0, z_0=1.0
    )

    # Verify that they are very close
    # Non-coherent power matches within 1.0 dB
    chex.assert_trees_all_close(
        power_non_coherent_differt,
        power_non_coherent_sionna,
        atol=1.0,
    )
    # Coherent power matches within 2.0 dB
    chex.assert_trees_all_close(
        power_coherent_differt,
        power_coherent_sionna,
        atol=2.0,
    )
