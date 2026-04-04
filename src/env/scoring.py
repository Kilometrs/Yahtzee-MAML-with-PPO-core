"""JAX-compatible Yahtzee scoring functions.

All functions are pure and jit-compatible. They operate on jnp arrays
and use no Python control flow.
"""
import jax
import jax.numpy as jnp


def _dice_counts(dice: jnp.ndarray) -> jnp.ndarray:
    """Count occurrences of each face value 0-6. Index 0 is unused."""
    return jnp.zeros(7, dtype=jnp.int32).at[dice].add(1)


def _score_upper(dice: jnp.ndarray, face: int) -> jnp.int32:
    """Score for upper section (Ones through Sixes)."""
    return jnp.sum(dice == face) * face


def _score_three_of_a_kind(dice: jnp.ndarray) -> jnp.int32:
    counts = _dice_counts(dice)
    return jnp.where(jnp.max(counts) >= 3, jnp.sum(dice), 0)


def _score_four_of_a_kind(dice: jnp.ndarray) -> jnp.int32:
    counts = _dice_counts(dice)
    return jnp.where(jnp.max(counts) >= 4, jnp.sum(dice), 0)


def _score_full_house(dice: jnp.ndarray) -> jnp.int32:
    counts = _dice_counts(dice)
    nonzero_counts = jnp.sort(counts)[::-1][:2]
    is_full_house = (nonzero_counts[0] == 3) & (nonzero_counts[1] == 2)
    return jnp.where(is_full_house, 25, 0)


def _score_small_straight(dice: jnp.ndarray) -> jnp.int32:
    present = jnp.zeros(7, dtype=jnp.bool_).at[dice].set(True)
    run_1234 = present[1] & present[2] & present[3] & present[4]
    run_2345 = present[2] & present[3] & present[4] & present[5]
    run_3456 = present[3] & present[4] & present[5] & present[6]
    return jnp.where(run_1234 | run_2345 | run_3456, 30, 0)


def _score_large_straight(dice: jnp.ndarray) -> jnp.int32:
    present = jnp.zeros(7, dtype=jnp.bool_).at[dice].set(True)
    low = present[1] & present[2] & present[3] & present[4] & present[5]
    high = present[2] & present[3] & present[4] & present[5] & present[6]
    return jnp.where(low | high, 40, 0)


def _score_yahtzee(dice: jnp.ndarray) -> jnp.int32:
    return jnp.where(jnp.all(dice == dice[0]), 50, 0)


def _score_chance(dice: jnp.ndarray) -> jnp.int32:
    return jnp.sum(dice)


_SCORE_FNS = [
    lambda d: _score_upper(d, 1),
    lambda d: _score_upper(d, 2),
    lambda d: _score_upper(d, 3),
    lambda d: _score_upper(d, 4),
    lambda d: _score_upper(d, 5),
    lambda d: _score_upper(d, 6),
    _score_three_of_a_kind,
    _score_four_of_a_kind,
    _score_full_house,
    _score_small_straight,
    _score_large_straight,
    _score_yahtzee,
    _score_chance,
]


def score_category(dice: jnp.ndarray, category: int) -> jnp.int32:
    """Score dice for a single category. Pure, jit-compatible."""
    return jax.lax.switch(category, _SCORE_FNS, dice)


def compute_all_scores(dice: jnp.ndarray) -> jnp.ndarray:
    """Compute scores for all 13 categories. Returns shape (13,)."""
    return jnp.array([score_category(dice, i) for i in range(13)])


def compute_valid_actions(
    dice,
    filled_mask,
):
    """Return all valid (category, column) pairs given current dice and filled slots.

    Placement hierarchy:
    1. Any unfilled slot where score_category(dice, category) > 0
    2. If none: any unfilled slot (cross-out, score = 0)

    Args:
        dice: Current dice values (list of 5 ints 1-6 or jnp array).
        filled_mask: Shape [n_columns][n_categories]. filled_mask[col][cat] = True if filled.

    Returns:
        List of (category_index, column_index) tuples representing valid actions.
    """
    from src.env.constants import N_CATEGORIES as _N_CATEGORIES
    import numpy as _np

    n_cols = len(filled_mask)
    n_cats = len(filled_mask[0]) if n_cols > 0 else _N_CATEGORIES

    dice_arr = jnp.array(dice, dtype=jnp.int32)

    positive = [
        (cat, col)
        for col in range(n_cols)
        for cat in range(n_cats)
        if not filled_mask[col][cat] and int(score_category(dice_arr, cat)) > 0
    ]
    if positive:
        return positive

    return [
        (cat, col)
        for col in range(n_cols)
        for cat in range(n_cats)
        if not filled_mask[col][cat]
    ]
