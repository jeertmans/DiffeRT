import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from typeguard import typechecked as typechecker


@jaxtyped
@typechecker
def distance(x: Float[Array, "m 3"], y: Float[Array, "n 3"]) -> Float[Array, "m n"]:
    return x


if __name__ == "__main__":
    x = jnp.ones((10, 3))
    y = jnp.ones((6, 3))
    print(distance(x, y))
