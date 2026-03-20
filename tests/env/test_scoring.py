from env.constants import (
    N_DICE, N_SIDES, N_CATEGORIES, MAX_REROLLS,
    UPPER_BONUS_THRESHOLD, UPPER_BONUS_VALUE, YAHTZEE_BONUS_VALUE,
    ONES, TWOS, THREES, FOURS, FIVES, SIXES,
    THREE_OF_A_KIND, FOUR_OF_A_KIND, FULL_HOUSE,
    SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE,
    CATEGORY_NAMES,
)
from env.scoring import score_category, compute_valid_actions


def test_constants_are_defined():
    assert N_DICE == 5
    assert N_SIDES == 6
    assert N_CATEGORIES == 13
    assert MAX_REROLLS == 2
    assert len(CATEGORY_NAMES) == 13


def test_score_category_is_callable():
    result = score_category([1, 1, 2, 3, 4], ONES)
    assert result is None or isinstance(result, int)


def test_compute_valid_actions_is_callable():
    filled = [[False] * 13 for _ in range(3)]
    result = compute_valid_actions([1, 2, 3, 4, 5], filled)
    assert result is None or isinstance(result, list)
