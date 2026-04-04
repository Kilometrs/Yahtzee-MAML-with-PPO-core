"""Tests for JAX reward task functions."""
import jax
import jax.numpy as jnp
import pytest
from src.tasks.reward_tasks import (
    TASK_NAMES, reward_max_score, reward_threshold_beater,
    reward_upper_bonus, reward_yahtzee_hunter, reward_conservative,
    compute_reward,
)
from src.env.constants import YAHTZEE


class TestTaskRegistry:
    def test_task_names(self):
        assert TASK_NAMES == [
            "MaxScore", "ThresholdBeater", "UpperBonus",
            "YahtzeeHunter", "Conservative",
        ]

    def test_task_count(self):
        assert len(TASK_NAMES) == 5


class TestMaxScore:
    def test_positive_delta(self):
        r = reward_max_score(prev_total=100, new_total=120, category=0,
                             score_gained=20, done=False, final_score=0,
                             yahtzee_bonus_delta=0, upper_crossed_63=False)
        assert float(r) == pytest.approx(20.0 / 50.0)

    def test_zero_delta(self):
        r = reward_max_score(prev_total=100, new_total=100, category=0,
                             score_gained=0, done=False, final_score=0,
                             yahtzee_bonus_delta=0, upper_crossed_63=False)
        assert float(r) == 0.0


class TestThresholdBeater:
    def test_above_threshold_at_done(self):
        r = reward_threshold_beater(prev_total=0, new_total=0, category=0,
                                    score_gained=0, done=True, final_score=300,
                                    yahtzee_bonus_delta=0, upper_crossed_63=False,
                                    threshold=250)
        assert float(r) == 1.0

    def test_below_threshold_at_done(self):
        r = reward_threshold_beater(prev_total=0, new_total=0, category=0,
                                    score_gained=0, done=True, final_score=200,
                                    yahtzee_bonus_delta=0, upper_crossed_63=False,
                                    threshold=250)
        assert float(r) == 0.0

    def test_not_done(self):
        r = reward_threshold_beater(prev_total=0, new_total=0, category=0,
                                    score_gained=0, done=False, final_score=300,
                                    yahtzee_bonus_delta=0, upper_crossed_63=False,
                                    threshold=250)
        assert float(r) == 0.0


class TestUpperBonus:
    def test_upper_category_positive_score(self):
        r = reward_upper_bonus(prev_total=0, new_total=0, category=2,
                               score_gained=9, done=False, final_score=0,
                               yahtzee_bonus_delta=0, upper_crossed_63=False)
        assert float(r) == pytest.approx(9.0 / 3.0)

    def test_upper_category_crossed_63(self):
        r = reward_upper_bonus(prev_total=0, new_total=0, category=2,
                               score_gained=9, done=False, final_score=0,
                               yahtzee_bonus_delta=0, upper_crossed_63=True)
        assert float(r) == pytest.approx(9.0 / 3.0 + 35.0)

    def test_lower_category_no_reward(self):
        r = reward_upper_bonus(prev_total=0, new_total=0, category=8,
                               score_gained=25, done=False, final_score=0,
                               yahtzee_bonus_delta=0, upper_crossed_63=False)
        assert float(r) == 0.0


class TestYahtzeeHunter:
    def test_yahtzee_scored(self):
        r = reward_yahtzee_hunter(prev_total=0, new_total=0, category=YAHTZEE,
                                  score_gained=50, done=False, final_score=0,
                                  yahtzee_bonus_delta=0, upper_crossed_63=False)
        assert float(r) == 50.0

    def test_yahtzee_bonus(self):
        r = reward_yahtzee_hunter(prev_total=0, new_total=0, category=3,
                                  score_gained=12, done=False, final_score=0,
                                  yahtzee_bonus_delta=100, upper_crossed_63=False)
        assert float(r) == 100.0

    def test_non_yahtzee(self):
        r = reward_yahtzee_hunter(prev_total=0, new_total=0, category=3,
                                  score_gained=12, done=False, final_score=0,
                                  yahtzee_bonus_delta=0, upper_crossed_63=False)
        assert float(r) == 0.0


class TestConservative:
    def test_positive_score(self):
        r = reward_conservative(prev_total=0, new_total=0, category=0,
                                score_gained=5, done=False, final_score=0,
                                yahtzee_bonus_delta=0, upper_crossed_63=False)
        assert float(r) == 1.0

    def test_zero_score_crossout(self):
        r = reward_conservative(prev_total=0, new_total=0, category=0,
                                score_gained=0, done=False, final_score=0,
                                yahtzee_bonus_delta=0, upper_crossed_63=False)
        assert float(r) == -1.0


class TestComputeReward:
    def test_dispatch_by_task_id(self):
        kwargs = dict(prev_total=100, new_total=120, category=0,
                      score_gained=20, done=False, final_score=0,
                      yahtzee_bonus_delta=0, upper_crossed_63=False,
                      threshold=250)
        r0 = compute_reward(task_id=0, **kwargs)
        assert float(r0) == pytest.approx(20.0 / 50.0)

    def test_is_jittable(self):
        jitted = jax.jit(compute_reward, static_argnums=())
        r = jitted(task_id=4, prev_total=0, new_total=0, category=0,
                   score_gained=5, done=False, final_score=0,
                   yahtzee_bonus_delta=0, upper_crossed_63=False,
                   threshold=250)
        assert float(r) == 1.0
