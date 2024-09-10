"""Utilities for working with 3D geometries."""

from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, ArrayLike, Float, jaxtyped


@jax.jit
@jaxtyped(typechecker=typechecker)
def pairwise_cross(
    u: Float[Array, "m 3"], v: Float[Array, "n 3"]
) -> Float[Array, "m n 3"]:
    """
    Compute the pairwise cross product between two arrays of vectors.

    Args:
        u: First array of vectors.
        v: Second array of vectors.

    Return:
        A 3D tensor with all cross products.
    """
    return jnp.cross(u[:, None, :], v[None, :, :])


@jax.jit
@jaxtyped(typechecker=typechecker)
def normalize(
    vector: Float[Array, "*batch 3"],
) -> tuple[Float[Array, "*batch 3"], Float[Array, " *batch"]]:
    """
    Normalize vectors and also return their length.

    This function avoids division by zero by checking vectors
    with zero-length, and returning unit length instead.

    Args:
        vector: An array of vectors.

    Return:
        The normalized vector and their length.

    Examples:
        The following examples shows how normalization works and
        its special behavior at zero.

        >>> from differt.geometry.utils import (
        ...     normalize,
        ... )
        >>>
        >>> vector = jnp.array([1.0, 1.0, 1.0])
        >>> normalize(vector)  # [1., 1., 1.] / sqrt(3), sqrt(3)
        (Array([0.5773503, 0.5773503, 0.5773503], dtype=float32),
         Array(1.7320508, dtype=float32))
        >>> zero = jnp.array([0.0, 0.0, 0.0])
        >>> normalize(zero)  # Special behavior at 0.
        (Array([0., 0., 0.], dtype=float32), Array(1., dtype=float32))
    """
    length: Array = jnp.linalg.norm(vector, axis=-1, keepdims=True)
    length = jnp.where(length == 0.0, jnp.ones_like(length), length)

    return vector / length, jnp.squeeze(length, axis=-1)


@partial(jax.jit, static_argnames=("normalize",))
@jaxtyped(typechecker=typechecker)
def orthogonal_basis(
    u: Float[Array, "*batch 3"], normalize: bool = True
) -> tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]]:
    """
    Generate ``v`` and ``w``, two other arrays of unit vectors that form with input ``u`` an orthogonal basis.

    Args:
        u: The first direction of the orthogonal basis.
            It must have a unit length.
        normalize: Whether the output vectors should be normalized.

            This may be needed, especially for vector ``v``,
            as floating-point error can accumulate so much
            that the vector lengths may diverge from the unit
            length by 10% or even more!

    Return:
        A pair of unit vectors, ``v`` and ``w``.

    Examples:
        The following example shows how this function works on basic input vectors.

        >>> from differt.geometry.utils import (
        ...     normalize,
        ...     orthogonal_basis,
        ... )
        >>>
        >>> u = jnp.array([1.0, 0.0, 0.0])
        >>> orthogonal_basis(u)
        (Array([ 0., -1.,  0.], dtype=float32), Array([ 0.,  0., -1.], dtype=float32))
        >>> u, _ = normalize(jnp.array([1.0, 1.0, 1.0]))
        >>> orthogonal_basis(u)
        (Array([ 0.4082483, -0.8164966,  0.4082483], dtype=float32),
         Array([ 0.7071068,  0.       , -0.7071068], dtype=float32))
    """
    vp = jnp.stack((u[..., 2], -u[..., 0], u[..., 1]), axis=-1)
    w = jnp.cross(u, vp, axis=-1)
    v = jnp.cross(w, u, axis=-1)

    if normalize:
        v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
        w = w / jnp.linalg.norm(w, axis=-1, keepdims=True)

    return v, w


@jax.jit
@jaxtyped(typechecker=typechecker)
def path_lengths(
    paths: Float[Array, "*batch path_length 3"],
) -> Float[Array, " *batch"]:
    """
    Compute the path length of each path.

    Each path is exactly made of ``path_length`` vertices.

    Args:
        paths: The array of path vertices.

    Return:
        The array of path lengths.

    Examples:
        The following example shows how to compute the length of a very simple path.

        >>> from differt.geometry.utils import (
        ...     path_lengths,
        ... )
        >>>
        >>> path = jnp.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        >>> path_lengths(path)
        Array(1., dtype=float32)
        >>> path_lengths(jnp.vstack((path, path[::-1, :])))
        Array(2., dtype=float32)
    """
    vectors = jnp.diff(paths, axis=-2)
    lengths = jnp.linalg.norm(vectors, axis=-1)

    return jnp.sum(lengths, axis=-1)


@jax.jit
@jaxtyped(typechecker=typechecker)
def rotation_matrix_along_x_axis(
    angle: Float[ArrayLike, " "],
) -> Float[Array, "3 3"]:
    """
    Return a rotation matrix to rotate coordinates along x axis.

    Args:
        angle: The rotation angle, in radians.

    Return:
        The rotation matrix.

    Examples:
        The following example shows how to rotate xyz coordinates
        along the x axis.

        >>> from differt.geometry.utils import (
        ...     rotation_matrix_along_x_axis,
        ... )
        >>>
        >>> xyz = jnp.array([1.0, 1.0, 0.0])
        >>> rotation_matrix_along_x_axis(jnp.pi / 2) @ xyz
        Array([ 1., -0.,  1.], dtype=float32)
    """
    co = jnp.cos(angle)
    si = jnp.sin(angle)

    return jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, +co, -si],
            [0.0, +si, +co],
        ]
    )


@jax.jit
@jaxtyped(typechecker=typechecker)
def rotation_matrix_along_y_axis(
    angle: Float[ArrayLike, " "],
) -> Float[Array, "3 3"]:
    """
    Return a rotation matrix to rotate coordinates along y axis.

    Args:
        angle: The rotation angle, in radians.

    Return:
        The rotation matrix.

    Examples:
        The following example shows how to rotate xyz coordinates
        along the y axis.

        >>> from differt.geometry.utils import (
        ...     rotation_matrix_along_y_axis,
        ... )
        >>>
        >>> xyz = jnp.array([1.0, 1.0, 0.0])
        >>> rotation_matrix_along_y_axis(jnp.pi / 2) @ xyz
        Array([-0.,  1., -1.], dtype=float32)
    """
    co = jnp.cos(angle)
    si = jnp.sin(angle)

    return jnp.array(
        [
            [+co, 0.0, +si],
            [0.0, 1.0, 0.0],
            [-si, 0.0, +co],
        ]
    )


@jax.jit
@jaxtyped(typechecker=typechecker)
def rotation_matrix_along_z_axis(
    angle: Float[ArrayLike, " "],
) -> Float[Array, "3 3"]:
    """
    Return a rotation matrix to rotate coordinates along z axis.

    Args:
        angle: The rotation angle, in radians.

    Return:
        The rotation matrix.

    Examples:
        The following example shows how to rotate xyz coordinates
        along the z axis.

        >>> from differt.geometry.utils import (
        ...     rotation_matrix_along_z_axis,
        ... )
        >>>
        >>> xyz = jnp.array([1.0, 0.0, 1.0])
        >>> rotation_matrix_along_z_axis(jnp.pi / 2) @ xyz
        Array([-0.,  1.,  1.], dtype=float32)
    """
    co = jnp.cos(angle)
    si = jnp.sin(angle)

    return jnp.array(
        [
            [+co, -si, 0.0],
            [+si, +co, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


@jax.jit
@jaxtyped(typechecker=typechecker)
def rotation_matrix_along_axis(
    angle: Float[ArrayLike, " "],
    axis: Float[Array, "3"],
) -> Float[Array, "3 3"]:
    """
    Return a rotation matrix to rotate coordinates along a given axis.

    Args:
        angle: The rotation angle, in radians.
        axis: A unit vector pointing in the axis' direction.

    Return:
        The rotation matrix.

    Examples:
        The following example shows how to rotate xyz coordinates
        along a given axis.

        >>> from differt.geometry.utils import (
        ...     normalize,
        ...     rotation_matrix_along_axis,
        ... )
        >>>
        >>> xyz = jnp.array([1.0, 1.0, 1.0])
        >>> axis, _ = normalize(jnp.array([1.0, 1.0, 0.0]))
        >>> rotation_matrix_along_axis(jnp.pi / 2, axis) @ xyz
        Array([ 1.7071066,  0.2928931, -0.       ], dtype=float32)

        In the following example, we show the importance of using a unit
        vector.

        >>> from differt.geometry.utils import (
        ...     rotation_matrix_along_axis,
        ... )
        >>>
        >>> xyz = jnp.array([1.0, 0.0, 1.0])
        >>> axis = jnp.array([1.0, 0.0, 0.0])
        >>> rotation_matrix_along_axis(jnp.pi, axis) @ xyz
        Array([ 1.       ,  0.0000001, -1.       ], dtype=float32)
        >>> axis = jnp.array([2.0, 0.0, 0.0])
        >>> rotation_matrix_along_axis(jnp.pi, axis) @ xyz
        Array([ 7.       ,  0.0000002, -1.       ], dtype=float32)
    """
    co = jnp.cos(angle)
    si = jnp.sin(angle)
    i = jnp.identity(3, dtype=axis.dtype)
    # Cross product matrix
    x = jnp.array(
        [
            [+0.0, -axis[2], +axis[1]],
            [+axis[2], +0.0, -axis[0]],
            [-axis[1], +axis[0], 0.0],
        ],
    )
    # Outer product matrix
    o = jnp.outer(axis, axis)

    return co * i + si * x + (1 - co) * o
