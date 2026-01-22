import chex
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import PRNGKeyArray

from differt.geometry import (
    rotation_matrix_along_x_axis,
    rotation_matrix_along_y_axis,
    rotation_matrix_along_z_axis,
)
from differt.utils import (
    sample_points_in_bounding_box,
)

from .generators import random_scene
from .utils import geometric_transformation, unpack_scene


@pytest.mark.parametrize("degenerate", [False, True])
def test_geometric_transformation_invariance_properties(
    degenerate: bool,
    key: PRNGKeyArray,
) -> None:
    scene_key, transform_key = jr.split(key, 2)

    # 1 - Generate random scene
    scene = random_scene(key=scene_key)

    if degenerate:
        # Set RX-TX aligned with z-axis
        scene = eqx.tree_at(
            lambda s: s.receivers,
            scene,
            scene.transmitters.at[-1].set(scene.receivers[-1]),
        )

    # 2 - Compute embeddings
    xyz = geometric_transformation(*unpack_scene(scene))

    # 3 - Apply random translation, scaling, and rotation to scene
    t_key, s_key, r_key = jr.split(transform_key, 3)

    # 3a - Test Translation Invariance
    t = sample_points_in_bounding_box(scene.mesh.bounding_box, key=t_key)
    scene_t = scene.translate(t)
    xyz_t = geometric_transformation(*unpack_scene(scene_t))
    chex.assert_trees_all_close(
        xyz, xyz_t, atol=1e-5, custom_message="Translation invariance failed"
    )

    # 3b - Test Scaling Invariance
    s = jnp.exp(jr.normal(s_key))
    scene_s = scene.scale(s)
    xyz_s = geometric_transformation(*unpack_scene(scene_s))
    chex.assert_trees_all_close(
        xyz, xyz_s, atol=1e-5, custom_message="Scaling invariance failed"
    )

    # 3c - Test Rotation Invariance (along z-axis)
    rot_z_angle = jr.uniform(r_key, minval=-jnp.pi, maxval=jnp.pi)
    R = rotation_matrix_along_z_axis(rot_z_angle)

    scene_r = scene.rotate(R)
    xyz_r = geometric_transformation(*unpack_scene(scene_r))

    chex.assert_trees_all_close(
        xyz, xyz_r, atol=1e-5, custom_message="Rotation invariance failed"
    )

    # 3d - Test Combined Transformations
    scene_combined = scene.translate(t).scale(s).rotate(R)
    xyz_combined = geometric_transformation(*unpack_scene(scene_combined))
    chex.assert_trees_all_close(
        xyz,
        xyz_combined,
        atol=1e-5,
        custom_message="Combined transformations invariance failed",
    )

    # 4 - Rotation invariance around x- and y-axes: expected failures
    rot_y_angle = jr.uniform(r_key, minval=-jnp.pi, maxval=jnp.pi)
    R = rotation_matrix_along_y_axis(rot_y_angle)

    scene_r = scene.rotate(R)
    xyz_r = geometric_transformation(*unpack_scene(scene_r))

    with pytest.raises(AssertionError, match="Rotation invariance failed"):
        chex.assert_trees_all_close(
            xyz, xyz_r, atol=1e-5, custom_message="Rotation invariance failed"
        )

    rot_x_angle = jr.uniform(r_key, minval=-jnp.pi, maxval=jnp.pi)
    R = rotation_matrix_along_x_axis(rot_x_angle)

    scene_r = scene.rotate(R)
    xyz_r = geometric_transformation(*unpack_scene(scene_r))

    with pytest.raises(AssertionError, match="Rotation invariance failed"):
        chex.assert_trees_all_close(
            xyz, xyz_r, atol=1e-5, custom_message="Rotation invariance failed"
        )
