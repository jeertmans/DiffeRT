import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from jaxtyping import PRNGKeyArray

from differt.geometry import TriangleMesh
from differt.scene import TriangleScene

from .agent import Agent
from .model import Model


@pytest.fixture(params=[1, 2])
def order(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def num_embeddings() -> int:
    return 32


@pytest.fixture
def width_size(num_embeddings: int) -> int:
    return 2 * num_embeddings


@pytest.fixture
def depth() -> int:
    return 3


@pytest.fixture
def dropout_rate() -> float:
    return 0.05


@pytest.fixture
def model(
    order: int,
    num_embeddings: int,
    width_size: int,
    depth: int,
    dropout_rate: float,
    key: PRNGKeyArray,
) -> Model:
    return Model(
        order=order,
        num_embeddings=num_embeddings,
        width_size=width_size,
        depth=depth,
        dropout_rate=dropout_rate,
        key=key,
    )


@pytest.fixture
def batch_size() -> int:
    return 64


@pytest.fixture
def optim() -> optax.GradientTransformationExtraArgs:
    return optax.adam(3e-4)


@pytest.fixture
def agent(
    model: Model,
    batch_size: int,
    optim: optax.GradientTransformationExtraArgs,
) -> Agent:
    return Agent(
        model=model,
        batch_size=batch_size,
        optim=optim,
    )


@pytest.fixture
def scene() -> TriangleScene:
    r = 1.0 / jnp.sqrt(3)
    vertices = jnp.array([
        [0.0, r, 0.0],  # Top vertex
        [-0.5, -0.5 * r, 0.0],  # Bottom left
        [0.5, -0.5 * r, 0.0],  # Bottom right
    ])
    triangles = jnp.array([
        [0, 1, 2],
    ])

    triangle = TriangleMesh(vertices=vertices, triangles=triangles)

    mesh = sum(
        (triangle.translate(jnp.array([0, 0, dz])) for dz in [-2, -1, 1, 2]),
        start=TriangleMesh.empty(),
    )

    return TriangleScene(
        transmitters=jnp.array([0, 0, +0.5]),
        receivers=jnp.array([0, 0, -0.5]),
        mesh=mesh,
    )


class TestAgent:
    def test_train(
        self, agent: Agent, scene: TriangleScene, key: PRNGKeyArray
    ) -> None:
        train_key, eval_key = jr.split(key)

        for key in jr.split(train_key, 10):
            agent, loss, avg_reward = agent.train(scene=scene, key=key)

        unreachable_objects = jnp.array([0, 3])
        path_candidates = jax.vmap(
            lambda key: agent.model(scene, inference=True, key=key)
        )(jr.split(eval_key, 10))

        paths = scene.compute_paths(order=agent.model.order)
        valid_paths = paths.masked()

        assert not jnp.isin(path_candidates, unreachable_objects).any(), (
            f"Path candidates should not contain unreachable objects, got: {path_candidates} (accepted are: {valid_paths.objects[:, 1:-1]})"
        )

        assert loss < 1.0
