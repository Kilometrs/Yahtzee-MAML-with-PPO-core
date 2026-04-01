"""Scoring logic for all 13 Yahtzee categories."""

from collections import Counter

from env.constants import (
    ONES, TWOS, THREES, FOURS, FIVES, SIXES,
    THREE_OF_A_KIND, FOUR_OF_A_KIND, FULL_HOUSE,
    SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE,
    N_CATEGORIES,
)


def score_category(dice: list[int], category: int) -> int:
    """Score a set of dice for the given category index.

    Returns the points earned. Returns 0 if dice do not satisfy the
    category requirements. Chance always returns sum of all dice.

    Args:
        dice: List of 5 integers (1-6).
        category: Category index (see constants.py).

    Returns:
        Integer score (>= 0).
    """
    counts = Counter(dice)

    if category == ONES:
        return counts[1]
    if category == TWOS:
        return counts[2] * 2
    if category == THREES:
        return counts[3] * 3
    if category == FOURS:
        return counts[4] * 4
    if category == FIVES:
        return counts[5] * 5
    if category == SIXES:
        return counts[6] * 6
    if category == THREE_OF_A_KIND:
        return sum(dice) if max(counts.values()) >= 3 else 0
    if category == FOUR_OF_A_KIND:
        return sum(dice) if max(counts.values()) >= 4 else 0
    if category == FULL_HOUSE:
        return 25 if sorted(counts.values()) == [2, 3] else 0
    if category == SMALL_STRAIGHT:
        s = set(dice)
        return 30 if ({1, 2, 3, 4} <= s or {2, 3, 4, 5} <= s or {3, 4, 5, 6} <= s) else 0
    if category == LARGE_STRAIGHT:
        s = set(dice)
        return 40 if s in ({1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}) else 0
    if category == YAHTZEE:
        return 50 if len(counts) == 1 else 0
    if category == CHANCE:
        return sum(dice)
    raise ValueError(f"Unknown category index: {category}")


def compute_valid_actions(
    dice: list[int],
    filled_mask: list[list[bool]],
) -> list[tuple[int, int]]:
    """Return all valid (category, column) pairs given current dice and filled slots.

    Placement hierarchy:
    1. Any unfilled slot where score_category(dice, category) > 0
    2. If none: any unfilled slot (cross-out, score = 0)

    Args:
        dice: Current dice values (list of 5 ints, 1-6).
        filled_mask: Shape [n_columns][n_categories]. filled_mask[col][cat] = True if filled.

    Returns:
        List of (category_index, column_index) tuples representing valid actions.
        Returns an empty list when all slots are filled (terminal state).
        Callers must check termination separately before acting on the result.
    """
    n_cols = len(filled_mask)
    n_cats = len(filled_mask[0]) if n_cols > 0 else N_CATEGORIES

    positive = [
        (cat, col)
        for col in range(n_cols)
        for cat in range(n_cats)
        if not filled_mask[col][cat] and score_category(dice, cat) > 0
    ]
    if positive:
        return positive

    # Forced cross-out: no positive-score slot exists anywhere
    return [
        (cat, col)
        for col in range(n_cols)
        for cat in range(n_cats)
        if not filled_mask[col][cat]
    ]
