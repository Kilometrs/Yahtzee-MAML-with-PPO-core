"""Scoring logic for all 13 Yahtzee categories."""

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
    pass


def compute_valid_actions(
    dice: list[int],
    filled_mask: list[list[bool]],
) -> list[tuple[int, int]]:
    """Return all valid (category, column) pairs given current dice and filled slots.

    Placement hierarchy:
    1. Any unfilled slot where score_category(dice, category) > 0
    2. If none: any unfilled Chance slot (always scores > 0 unless all dice are 0,
       which cannot happen)
    3. If all Chance slots also filled: any unfilled slot (cross-out, score = 0)

    Args:
        dice: Current dice values (list of 5 ints, 1-6).
        filled_mask: filled_mask[col][cat] = True if that slot is already filled.

    Returns:
        List of (category_index, column_index) tuples representing valid actions.
    """
    pass
