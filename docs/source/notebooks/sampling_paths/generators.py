from collections.abc import Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Key,
    PRNGKeyArray,
)
from tqdm.auto import trange

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

    The number of objects is randomly sampled based on a random fill factor.

    Args:
        min_fill_factor: The minimum fill factor to be used.
        max_fill_factor: The maximum fill factor to be used.
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


def validation_scene_keys(
    *,
    order: int,
    num_scenes: int = 100,
    progress: bool = True,
    key: PRNGKeyArray,
) -> Key[Array, " num_scenes"]:
    """
    Return a fixed set of scene keys for validating the model.

    Args:
        order: The path order to be used.
        num_scenes: The number of scene keys to generate.
        progress: Whether to show a progress bar when generating the scenes.
        key: The random key to be used.
    Returns:
        A fixed set of scene keys, for which the corresponding scenes contain valid paths of the given order.
    """

    def keys(key: PRNGKeyArray) -> Iterator[PRNGKeyArray]:
        old_key = key
        while True:
            old_key, new_key = jr.split(old_key)
            yield new_key

    generator = filter(
        lambda key: (
            random_scene(key=key).compute_paths(order=order).mask.sum() > 0
        ),
        keys(key),
    )

    if progress:
        it = trange(num_scenes, desc="Selecting validation scenes")
    else:
        it = range(num_scenes)

    return jnp.stack([next(generator) for _ in it])
