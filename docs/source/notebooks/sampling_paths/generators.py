from collections.abc import Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    PRNGKeyArray,
)

from differt.scene import (
    TriangleScene,
    download_sionna_scenes,
    get_sionna_scene,
)
from differt.utils import sample_points_in_bounding_box

download_sionna_scenes()
BASE_SCENE = TriangleScene.load_xml(get_sionna_scene("simple_street_canyon"))


@eqx.filter_jit
def random_scene(
    *,
    min_fill_factor: float = 0.5,
    max_fill_factor: float = 1.0,
    key: PRNGKeyArray,
) -> TriangleScene:
    """
    Return a random scene with one TX and one RX, at random positions, and a random number of objects.

    The number of objects can be anywhere between 'num_objects // 2' and 'num_objects',
    where 'num_objects' is the total number of objects in the scene. Again, we avoid
    sampling scene with very few objects, as they usually contain no valid paths.

    Args:
        key: The random key to be used.

    Returns:
        A new scene.
    """
    key_tx, key_rx, key_fill_factor, key_sample_triangles = jr.split(key, 4)

    bounding_box = BASE_SCENE.mesh.bounding_box.at[:, :2].apply(
        lambda xy: jnp.clip(xy, -5.0, 5.0)
    )
    rx_bounding_box = bounding_box.at[:, 2].apply(
        lambda z: jnp.clip(z, 1.0, 2.0)
    )
    tx_bounding_box = bounding_box.at[:, 2].apply(
        lambda z: jnp.clip(z, 2.0, 50.0)
    )
    fill_factor = jr.uniform(
        key_fill_factor, (), minval=min_fill_factor, maxval=max_fill_factor
    )
    scene = eqx.tree_at(
        lambda s: (s.transmitters, s.receivers, s.mesh),
        BASE_SCENE,
        (
            sample_points_in_bounding_box(tx_bounding_box, key=key_tx),
            sample_points_in_bounding_box(rx_bounding_box, key=key_rx),
            BASE_SCENE.mesh.sample(
                fill_factor,
                by_masking=True,
                sample_objects=True,
                key=key_sample_triangles,
            ),
        ),
    )
    # Assumption: the floor is the only object that is made of 2 triangles
    object_bounds = BASE_SCENE.mesh.object_bounds
    assert object_bounds is not None
    is_floor = object_bounds[:, 0] + 2 == object_bounds[:, 1]
    floor_index = object_bounds[
        jnp.argwhere(is_floor, size=1, fill_value=-1)[0], 0
    ]
    scene = eqx.tree_at(
        lambda s: s.mesh.mask,
        scene,
        jax.lax.dynamic_update_slice(
            scene.mesh.mask,
            jnp.array([True, True]),
            floor_index,
            allow_negative_indices=False,
        ),
    )
    return scene


def train_dataloader(*, key: PRNGKeyArray) -> Iterator[TriangleScene]:
    """
    Return an (infinite) iterator over random scenes for training the model.

    Args:
        key: The random key to be used.

    Yields:
        An infinite number of random scenes.
    """

    while True:
        key, key_to_use = jr.split(key, 2)
        yield random_scene(key=key_to_use)
