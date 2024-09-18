import equinox as eqx
import numpy as np
import jax.numpy as jnp
from jaxtyping import jaxtyped, Float, Int, Array
from beartype import beartype as typechecker

from differt.plotting import draw_paths

@jaxtyped(typechecker=typechecker)
class Paths(eqx.Module):
    """
    A convenient wrapper class around path vertices and object indices.

    This class can hold arbitrary many paths, but they must share the same
    length, i.e., the same number of vertices per path.
    """
    vertices: Float[Array, "*batch path_length 3"]
    """The array of path vertices."""
    objects: Int[Array, "*batch path_length"]
    """The array of object indices.
    
    To every path vertex corresponds one object (e.g., a triangle).
    A placeholder value of :python:`-1` can be used in specific cases,
    like for transmitter and receiver positions.
    """
    @eqx.filter_jit
    def group_by_objects(self) -> Int[Array, " *batch"]:
        """
        Return an array of unique object groups.

        This function is useful to group paths that
        undergo the same types of interactions.

        Returns:
            An array of group indices.

        Examples:
            The following shows how one can group
            paths by object groups. There are two different objects,
            denoted by indices ``0`` and ``1``, and each path is made
            of three vertices.

            >>> from differt.geometry.paths import Paths
            >>>
            >>> key = jax.random.PRNGKey(1234)
            >>> key_v, key_o = jax.random.split(key, 2)
            >>> *batch, path_length = (2, 6, 3)
            >>> vertices = jax.random.uniform(key_v, (*batch, path_length, 3))
            >>> objects = jax.random.randint(key_o, (*batch, path_length), 0, 2)
            >>> objects
            Array([[[1, 0, 0],
                    [1, 0, 0],
                    [0, 1, 1],
                    [1, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1]],
                   [[1, 0, 1],
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 1, 0],
                    [0, 0, 1],
                    [0, 1, 1]]], dtype=int32)
            >>> paths = Paths(vertices, objects)
            >>> groups = paths.group_by_objects()
            >>> groups
            Array([[4, 4, 3, 5, 2, 3],
                   [5, 0, 0, 6, 1, 3]], dtype=int32)
        """
        *batch, path_length = self.objects.shape

        objects = self.objects.reshape((-1, path_length))
        inverse = jnp.unique(objects, axis=0, size=objects.shape[0], return_inverse=True)[1]

        return inverse.reshape(batch)


    def plot(self):
        return draw_paths(np.asarray(self.vertices))