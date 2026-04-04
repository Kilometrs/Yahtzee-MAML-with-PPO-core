"""Tests for env constants and scoring functions."""
import pytest
import jax
import jax.numpy as jnp
from src.env.constants import (
    N_DICE, N_SIDES, N_CATEGORIES, MAX_REROLLS,
    UPPER_BONUS_THRESHOLD, UPPER_BONUS_VALUE, YAHTZEE_BONUS_VALUE,
    ONES, TWOS, THREES, FOURS, FIVES, SIXES,
    THREE_OF_A_KIND, FOUR_OF_A_KIND, FULL_HOUSE,
    SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE,
    UPPER_CATEGORIES, PHASE_ROLL, PHASE_SCORE,
)
from src.env.scoring import score_category, compute_all_scores


class TestConstants:
    def test_basic_constants(self):
        assert N_DICE == 5
        assert N_SIDES == 6
        assert N_CATEGORIES == 13
        assert MAX_REROLLS == 2

    def test_bonus_constants(self):
        assert UPPER_BONUS_THRESHOLD == 63
        assert UPPER_BONUS_VALUE == 35
        assert YAHTZEE_BONUS_VALUE == 100

    def test_category_indices(self):
        assert ONES == 0
        assert CHANCE == 12
        assert YAHTZEE == 11
        assert len(UPPER_CATEGORIES) == 6

    def test_phase_constants(self):
        assert PHASE_ROLL == 0
        assert PHASE_SCORE == 1


class TestScoreCategory:
    def test_ones(self):
        dice = jnp.array([1, 1, 3, 4, 5], dtype=jnp.int32)
        assert score_category(dice, 0) == 2

    def test_sixes(self):
        dice = jnp.array([6, 6, 6, 1, 2], dtype=jnp.int32)
        assert score_category(dice, 5) == 18

    def test_three_of_a_kind_hit(self):
        dice = jnp.array([3, 3, 3, 4, 5], dtype=jnp.int32)
        assert score_category(dice, 6) == 18

    def test_three_of_a_kind_miss(self):
        dice = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
        assert score_category(dice, 6) == 0

    def test_four_of_a_kind_hit(self):
        dice = jnp.array([2, 2, 2, 2, 5], dtype=jnp.int32)
        assert score_category(dice, 7) == 13

    def test_four_of_a_kind_miss(self):
        dice = jnp.array([2, 2, 2, 3, 5], dtype=jnp.int32)
        assert score_category(dice, 7) == 0

    def test_full_house_hit(self):
        dice = jnp.array([2, 2, 5, 5, 5], dtype=jnp.int32)
        assert score_category(dice, 8) == 25

    def test_full_house_three_two(self):
        dice = jnp.array([2, 2, 2, 5, 5], dtype=jnp.int32)
        assert score_category(dice, 8) == 25

    def test_full_house_all_same_is_not_full_house(self):
        dice = jnp.array([3, 3, 3, 3, 3], dtype=jnp.int32)
        assert score_category(dice, 8) == 0

    def test_small_straight_1234(self):
        dice = jnp.array([1, 2, 3, 4, 6], dtype=jnp.int32)
        assert score_category(dice, 9) == 30

    def test_small_straight_2345(self):
        dice = jnp.array([2, 3, 4, 5, 1], dtype=jnp.int32)
        assert score_category(dice, 9) == 30

    def test_small_straight_3456(self):
        dice = jnp.array([3, 4, 5, 6, 1], dtype=jnp.int32)
        assert score_category(dice, 9) == 30

    def test_small_straight_miss(self):
        dice = jnp.array([1, 2, 4, 5, 6], dtype=jnp.int32)
        assert score_category(dice, 9) == 0

    def test_large_straight_low(self):
        dice = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
        assert score_category(dice, 10) == 40

    def test_large_straight_high(self):
        dice = jnp.array([2, 3, 4, 5, 6], dtype=jnp.int32)
        assert score_category(dice, 10) == 40

    def test_large_straight_miss(self):
        dice = jnp.array([1, 2, 3, 4, 6], dtype=jnp.int32)
        assert score_category(dice, 10) == 0

    def test_yahtzee_hit(self):
        dice = jnp.array([4, 4, 4, 4, 4], dtype=jnp.int32)
        assert score_category(dice, 11) == 50

    def test_yahtzee_miss(self):
        dice = jnp.array([4, 4, 4, 4, 3], dtype=jnp.int32)
        assert score_category(dice, 11) == 0

    def test_chance(self):
        dice = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
        assert score_category(dice, 12) == 15

    def test_is_jittable(self):
        dice = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
        jitted = jax.jit(score_category)
        assert jitted(dice, 0) == 1


class TestComputeAllScores:
    def test_returns_13_scores(self):
        dice = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
        scores = compute_all_scores(dice)
        assert scores.shape == (13,)

    def test_known_dice(self):
        dice = jnp.array([3, 3, 3, 3, 3], dtype=jnp.int32)
        scores = compute_all_scores(dice)
        assert scores[2] == 15   # threes: 5*3
        assert scores[6] == 15   # three of a kind: sum
        assert scores[7] == 15   # four of a kind: sum
        assert scores[8] == 0    # full house: no (all same)
        assert scores[11] == 50  # yahtzee
        assert scores[12] == 15  # chance

    def test_is_jittable(self):
        dice = jnp.array([1, 1, 1, 2, 2], dtype=jnp.int32)
        jitted = jax.jit(compute_all_scores)
        scores = jitted(dice)
        assert scores.shape == (13,)
