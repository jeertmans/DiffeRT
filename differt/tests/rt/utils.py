import jax.numpy as jnp


class PlanarMirrorsSetup:
    """
    Test setup that looks something like:

                1           3
             ───────     ───────
           0                     4
    (from) x                     x (to)

                   ───────
                      2

    where xs are starting and ending vertices, and '───────' are mirrors.
    """

    def __init__(self, *batch: int) -> None:
        from_vertex = jnp.array([0.0, 0.0, 0.0])
        to_vertex = jnp.array([1.0, 0.0, 0.0])
        mirror_vertices = jnp.array(
            [[0.0, +1.0, 0.0], [0.0, -1.0, 0.0], [0.0, +1.0, 0.0]],
        )
        mirror_normals = jnp.array(
            [[0.0, -1.0, 0.0], [0.0, +1.0, 0.0], [0.0, -1.0, 0.0]],
        )
        path = jnp.array(
            [[1.0 / 6.0, +1.0, 0.0], [3.0 / 6.0, -1.0, 0.0], [5.0 / 6.0, +1.0, 0.0]],
        )
        # Tile on batch dimensions
        axis = tuple(range(len(batch)))
        self.from_vertices = jnp.tile(from_vertex, (*batch, 1))
        assert self.from_vertices.shape == (*batch, 3)
        self.to_vertices = jnp.tile(to_vertex, (*batch, 1))
        assert self.to_vertices.shape == (*batch, 3)
        self.mirror_vertices = jnp.tile(
            jnp.expand_dims(mirror_vertices, axis),
            (*batch, 1, 1),
        )
        assert self.mirror_vertices.shape == (*batch, 3, 3)
        self.mirror_normals = jnp.tile(
            jnp.expand_dims(mirror_normals, axis),
            (*batch, 1, 1),
        )
        assert self.mirror_normals.shape == (*batch, 3, 3)
        self.paths = jnp.tile(jnp.expand_dims(path, axis), (*batch, 1, 1))
        assert self.paths.shape == (*batch, 3, 3)

        _ = jnp.concatenate(
            (
                jnp.expand_dims(self.from_vertices, -2),
                self.paths,
                jnp.expand_dims(self.to_vertices, -2),
            ),
            axis=-2,
        )  # Check we can concatenate
