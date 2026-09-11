import chex
import jax.numpy as jnp

from differt.geometry.solvers._dispatch import (
    _bending_first_permutation,
    _nearest_real_neighbor_indices,
)


def test_bending_first_permutation_moves_true_to_front() -> None:
    is_bending = jnp.array([False, True, False, True, True])
    perm = _bending_first_permutation(is_bending)
    reordered = is_bending[perm]
    chex.assert_trees_all_equal(reordered, jnp.array([True, True, True, False, False]))
    # Relative order preserved within each group.
    chex.assert_trees_all_equal(perm, jnp.array([1, 3, 4, 0, 2]))


def test_bending_first_permutation_all_true() -> None:
    is_bending = jnp.array([True, True, True])
    perm = _bending_first_permutation(is_bending)
    chex.assert_trees_all_equal(perm, jnp.array([0, 1, 2]))


def test_bending_first_permutation_all_false() -> None:
    is_bending = jnp.array([False, False, False])
    perm = _bending_first_permutation(is_bending)
    chex.assert_trees_all_equal(perm, jnp.array([0, 1, 2]))


def test_bending_first_permutation_batched() -> None:
    is_bending = jnp.array([
        [False, True, False, True, True],
        [True, True, True, False, False],
    ])
    perm = _bending_first_permutation(is_bending)
    reordered = jnp.take_along_axis(is_bending, perm, axis=-1)
    chex.assert_trees_all_equal(
        reordered,
        jnp.array([
            [True, True, True, False, False],
            [True, True, True, False, False],
        ]),
    )


def test_nearest_real_neighbor_indices_basic() -> None:
    # Positions: 0=real(tx), 1=fake, 2=real, 3=fake, 4=fake, 5=real(rx)
    is_real = jnp.array([True, False, True, False, False, True])
    prev_idx, next_idx = _nearest_real_neighbor_indices(is_real)
    chex.assert_trees_all_equal(prev_idx, jnp.array([0, 0, 2, 2, 2, 5]))
    chex.assert_trees_all_equal(next_idx, jnp.array([0, 2, 2, 5, 5, 5]))


def test_nearest_real_neighbor_indices_endpoints_only() -> None:
    is_real = jnp.array([True, False, False, True])
    prev_idx, next_idx = _nearest_real_neighbor_indices(is_real)
    chex.assert_trees_all_equal(prev_idx, jnp.array([0, 0, 0, 3]))
    chex.assert_trees_all_equal(next_idx, jnp.array([0, 3, 3, 3]))


def test_nearest_real_neighbor_indices_all_real() -> None:
    is_real = jnp.array([True, True, True])
    prev_idx, next_idx = _nearest_real_neighbor_indices(is_real)
    chex.assert_trees_all_equal(prev_idx, jnp.array([0, 1, 2]))
    chex.assert_trees_all_equal(next_idx, jnp.array([0, 1, 2]))


def test_nearest_real_neighbor_indices_batched() -> None:
    is_real = jnp.array([
        [True, False, True, False, False, True],
        [True, True, True, True, True, True],
    ])
    prev_idx, next_idx = _nearest_real_neighbor_indices(is_real)
    chex.assert_trees_all_equal(
        prev_idx, jnp.array([[0, 0, 2, 2, 2, 5], [0, 1, 2, 3, 4, 5]])
    )
    chex.assert_trees_all_equal(
        next_idx, jnp.array([[0, 2, 2, 5, 5, 5], [0, 1, 2, 3, 4, 5]])
    )
