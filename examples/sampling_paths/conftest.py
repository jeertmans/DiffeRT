import jax
import jax.numpy as jnp
import optax
import pytest
from jaxtyping import PRNGKeyArray

from differt.geometry import TriangleMesh
from differt.scene import TriangleScene

from .agent import Agent
from .model import Model
from .submodels import Flows, ObjectsEncoder, SceneEncoder, StateEncoder


@pytest.fixture
def seed() -> int:
    return 1234


@pytest.fixture
def key(seed: int) -> PRNGKeyArray:
    return jax.random.key(seed)


@pytest.fixture(params=[1, 2, 3])
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
    return 2


@pytest.fixture
def dropout_rate() -> float:
    return 0.0

@pytest.fixture
def epsilon() -> float:
    return 0.1


@pytest.fixture
def model(
    order: int,
    num_embeddings: int,
    width_size: int,
    depth: int,
    dropout_rate: float,
    epsilon: float,
    key: PRNGKeyArray,
) -> Model:
    return Model(
        order=order,
        num_embeddings=num_embeddings,
        width_size=width_size,
        depth=depth,
        dropout_rate=dropout_rate,
        epsilon=epsilon,
        key=key,
    )


@pytest.fixture
def flows(model: Model) -> Flows:
    return model.flows


@pytest.fixture
def objects_encoder(model: Model) -> ObjectsEncoder:
    return model.objects_encoder


@pytest.fixture
def scene_encoder(model: Model) -> SceneEncoder:
    return model.scene_encoder


@pytest.fixture
def state_encoder(model: Model) -> StateEncoder:
    return model.state_encoder


@pytest.fixture
def batch_size() -> int:
    return 64


@pytest.fixture
def optim() -> optax.GradientTransformationExtraArgs:
    return optax.adam(3e-4)

@pytest.fixture
def delta_epsilon() -> float:
    return 0.0

@pytest.fixture
def min_epsilon(epsilon) -> float:
    return epsilon


@pytest.fixture
def agent(
    model: Model,
    batch_size: int,
    optim: optax.GradientTransformationExtraArgs,
    delta_epsilon: float,
    min_epsilon: float,
) -> Agent:
    return Agent(
        model=model,
        batch_size=batch_size,
        optim=optim,
        delta_epsilon=delta_epsilon,
        min_epsilon=min_epsilon,
    )


@pytest.fixture(params=[True, False])
def inference(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture
def scene(key: PRNGKeyArray) -> TriangleScene:
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

    # We add inactive objects to the scene
    # to make sure the agent effectively ignores them
    mesh += triangle.sample(0.0, by_masking=True, key=key)

    return TriangleScene(
        transmitters=jnp.array([0, 0, +0.5]),
        receivers=jnp.array([0, 0, -0.5]),
        mesh=mesh,
    )
