# ruff: noqa: ERA001
from collections.abc import Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from equinox import nn
from jaxtyping import Array, Float, PRNGKeyArray
from pytest_codspeed import BenchmarkFixture

from differt.scene._triangle_scene import TriangleScene
from differt.utils import sample_points_in_bounding_box


def random_scene(
    base_scene: TriangleScene, num_tx: int = 10, num_rx: int = 10, *, key: PRNGKeyArray
) -> TriangleScene:
    scene = base_scene
    key_tx, key_rx, key_num_objects, key_sample_triangles = jax.random.split(key, 4)
    bounding_box = scene.mesh.bounding_box
    scene = eqx.tree_at(
        lambda s: s.transmitters,
        scene,
        sample_points_in_bounding_box(bounding_box, (num_tx,), key=key_tx),
    )
    scene = eqx.tree_at(
        lambda s: s.receivers,
        scene,
        sample_points_in_bounding_box(bounding_box, (num_rx,), key=key_rx),
    )
    num_objects = scene.mesh.num_objects
    num_objects = jax.random.randint(key_num_objects, (), 0, num_objects + 1)
    return eqx.tree_at(
        lambda s: s.mesh,
        scene,
        scene.mesh.sample(int(num_objects), key=key_sample_triangles),
    )


def train_dataloader(
    base_scene: TriangleScene, *, key: PRNGKeyArray
) -> Iterator[TriangleScene]:
    while True:
        key, key_to_use = jax.random.split(key, 2)
        yield random_scene(base_scene, key=key_to_use)


class LOSModel(eqx.Module):
    embeds: nn.MLP
    logits: nn.Linear
    inference: bool

    def __init__(
        self,
        num_embeds: int = 32,
        width: int = 64,
        depth: int = 3,
        *,
        inference: bool = False,
        key: PRNGKeyArray,
    ) -> None:
        key_embeds, key_logits = jax.random.split(key)
        self.embeds = nn.MLP(
            in_size=3,
            out_size=num_embeds,
            width_size=width,
            depth=depth,
            key=key_embeds,
        )
        self.logits = nn.Linear(num_embeds * 3, "scalar", key=key_logits)
        self.inference = inference

    def __call__(
        self,
        triangle_vertices: Float[Array, "num_triangles 3 3"],
        path_vertices: Float[Array, "2 3"],
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, " "]:
        # [num_triangles 3 num_embeds] -> [num_triangles num_embeds]
        triangle_embeds = jax.vmap(jax.vmap(self.embeds))(triangle_vertices).mean(
            axis=1
        )

        if self.inference:
            where = jax.random.uniform(key, shape=(triangle_embeds.shape[0], 1)) < 0.1
            # [num_embeds]
            scene_embeds = triangle_embeds.sum(axis=0, where=where, initial=0.0)
        else:
            del key
            # [num_embeds]
            scene_embeds = triangle_embeds.sum(axis=0)

        # [2 num_embeds] -> [2*num_embeds]
        path_embeds = jax.vmap(self.embeds)(path_vertices).reshape(-1)

        logits = self.logits(jnp.concatenate([scene_embeds, path_embeds], axis=-1))
        return jax.nn.sigmoid(logits)


@eqx.filter_jit(donate="all-except-first")
def loss(model: LOSModel, scene: TriangleScene, key: PRNGKeyArray) -> Array:
    paths = scene.compute_paths(order=0)
    f = lambda *x: model(*x, key=key)  # noqa: E731

    for _ in range(paths.vertices.ndim - 2):
        f = jax.vmap(f, in_axes=(None, 0))

    pred = f(scene.mesh.triangle_vertices, paths.vertices)

    assert paths.mask is not None
    return jnp.mean((pred - paths.mask.astype(pred.dtype)) ** 2)


def train(
    model: LOSModel,
    dataloader: Iterator[TriangleScene],
    *,
    steps: int = 100,
    key: PRNGKeyArray,
) -> tuple[LOSModel, Float[Array, " "]]:
    optim = optax.adam(learning_rate=1e-3)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit(donate="all")
    def make_step(
        model: LOSModel,
        opt_state: optax.OptState,
        scene: TriangleScene,
        *,
        key: PRNGKeyArray,
    ) -> tuple[LOSModel, optax.OptState, Float[Array, " "]]:
        loss_value, grads = eqx.filter_value_and_grad(loss)(
            model,
            scene,
            key=key,
        )
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    loss_value = jnp.array(0.0)

    for key_step, scene in zip(jax.random.split(key, steps), dataloader, strict=False):
        model, opt_state, loss_value = make_step(model, opt_state, scene, key=key_step)

    return nn.inference_mode(model), loss_value


@pytest.mark.benchmark(group="los_model")
def test_train_time(
    simple_street_canyon_scene: TriangleScene,
    benchmark: BenchmarkFixture,
    key: PRNGKeyArray,
) -> None:
    def bench_fun() -> None:
        key_model, key_dataloader, key_train = jax.random.split(key, 3)
        model = LOSModel(key=key_model)
        dataloader = train_dataloader(simple_street_canyon_scene, key=key_dataloader)
        train(model, dataloader, key=key_train)[1].block_until_ready()

    _ = benchmark(bench_fun)
