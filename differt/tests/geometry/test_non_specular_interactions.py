"""End-to-end integration tests for non-specular path tracing.

Exercises 'Scene.trace_paths'/'Scene.trace_fields' with
'allowed_interactions' through the full pipeline: candidate generation,
mixed-interaction geometric solving, and EM field computation.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest

from differt.em import InteractionType, Material, materials
from differt.geometry._mesh import Mesh
from differt.geometry._scene import Scene
from differt.geometry.solvers._sbr import SBRPathTracer


@pytest.fixture
def wedge_scene() -> Scene:
    # A right-angle convex wedge (like a building corner), with exactly one
    # diffraction edge (the shared edge between the two triangles).
    vertices = jnp.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [1.0, 0.0, -1.0],  # 3
    ])
    triangles = jnp.array([
        [0, 1, 2],
        [1, 3, 2],
    ])
    mesh = Mesh(
        vertices=vertices,
        triangles=triangles,
        assume_quads=False,
        face_materials=jnp.array([0, 0]),
        material_names=("itu_concrete",),
    )
    return Scene(
        transmitters=jnp.array([0.5, 0.5, 1.0]),
        receivers=jnp.array([2.0, 0.5, -0.5]),
        mesh=mesh,
    )


@pytest.fixture
def wedge_and_wall_scene(wedge_scene: Scene) -> Scene:
    mesh = wedge_scene.mesh
    vertices = jnp.concatenate([
        mesh.vertices,
        jnp.array([
            [3.0, -1.0, -2.0],
            [3.0, 2.0, -2.0],
            [3.0, -1.0, 2.0],
        ]),
    ])
    triangles = jnp.concatenate([mesh.triangles, jnp.array([[4, 5, 6]])])
    face_materials = jnp.concatenate([mesh.face_materials, jnp.array([1])])
    new_mesh = Mesh(
        vertices=vertices,
        triangles=triangles,
        assume_quads=False,
        face_materials=face_materials,
        material_names=(*mesh.material_names, "vacuum_with_thickness"),
    )
    return Scene(
        transmitters=jnp.array([0.5, 0.5, 1.0]),
        receivers=jnp.array([5.0, 0.5, -0.5]),
        mesh=new_mesh,
    )


@pytest.fixture
def radio_materials_with_thickness() -> dict[str, Material]:
    vacuum = materials["itu_concrete"]  # any built-in, just for properties
    vacuum_with_thickness = Material(
        name="vacuum_with_thickness",
        properties=vacuum.properties,
        thickness=0.1,
    )
    concrete_with_thickness = Material(
        name="itu_concrete",
        properties=materials["itu_concrete"].properties,
        thickness=0.2,
    )
    # 'TRANSMISSION' is allowed on every primitive (not just the wall), so
    # every material that could be hit by a TRANSMISSION bounce needs a
    # finite thickness, including the wedge's own material.
    return {
        "itu_concrete": concrete_with_thickness,
        "vacuum_with_thickness": vacuum_with_thickness,
    }


def test_trace_paths_diffraction_via_scene(wedge_scene: Scene) -> None:
    paths = wedge_scene.trace_paths(
        order=1,
        solver="exhaustive",
        allowed_interactions=frozenset({InteractionType.DIFFRACTION}),
    ).masked()
    assert paths.num_valid_paths > 0
    assert jnp.all(paths.interaction_types[paths.mask] == InteractionType.DIFFRACTION)


def test_trace_fields_diffraction_via_scene(
    wedge_scene: Scene, radio_materials_with_thickness: dict[str, Material]
) -> None:
    fields = wedge_scene.trace_fields(
        order=1,
        frequency=1e9,
        path_solver="exhaustive",
        allowed_interactions=frozenset({InteractionType.DIFFRACTION}),
        field_solver_kwargs={"radio_materials": radio_materials_with_thickness},
    )
    valid = fields.mask
    assert jnp.any(valid)
    assert jnp.all(jnp.isfinite(fields.fields[valid]))


def test_trace_fields_transmission_via_scene(
    wedge_and_wall_scene: Scene, radio_materials_with_thickness: dict[str, Material]
) -> None:
    # A transmitter/receiver pair whose direct line of sight crosses the
    # wall (x=3) within its finite bounds, at y=0.5, z=-1.5.
    scene = eqx.tree_at(
        lambda s: (s.transmitters, s.receivers),
        wedge_and_wall_scene,
        (jnp.array([2.0, 0.5, -1.5]), jnp.array([4.0, 0.5, -1.5])),
    )
    fields = scene.trace_fields(
        order=1,
        frequency=1e9,
        path_solver="exhaustive",
        allowed_interactions=frozenset({InteractionType.TRANSMISSION}),
        field_solver_kwargs={"radio_materials": radio_materials_with_thickness},
    )
    valid = fields.mask
    assert jnp.any(valid)
    assert jnp.all(jnp.isfinite(fields.fields[valid]))


def test_trace_fields_diffraction_then_transmission_via_scene(
    wedge_and_wall_scene: Scene, radio_materials_with_thickness: dict[str, Material]
) -> None:
    fields = wedge_and_wall_scene.trace_fields(
        order=2,
        frequency=1e9,
        path_solver="exhaustive",
        allowed_interactions=frozenset({
            InteractionType.DIFFRACTION,
            InteractionType.TRANSMISSION,
        }),
        field_solver_kwargs={"radio_materials": radio_materials_with_thickness},
    )
    valid = fields.mask
    assert jnp.any(valid)
    assert jnp.all(jnp.isfinite(fields.fields[valid]))


def test_sbr_rejects_non_reflection_interactions(wedge_scene: Scene) -> None:
    with pytest.raises(NotImplementedError, match="only supports 'REFLECTION'"):
        wedge_scene.trace_paths(
            order=1,
            solver=SBRPathTracer(),
            allowed_interactions=frozenset({InteractionType.DIFFRACTION}),
        )


def test_scene_trace_paths_default_is_still_reflection_only(
    wedge_and_wall_scene: Scene,
) -> None:
    paths = wedge_and_wall_scene.trace_paths(order=1, solver="exhaustive").masked()
    if paths.num_valid_paths > 0:
        assert jnp.all(
            paths.interaction_types[paths.mask] == InteractionType.REFLECTION
        )
