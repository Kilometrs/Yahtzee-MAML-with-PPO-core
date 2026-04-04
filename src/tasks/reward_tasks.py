"""Pure JAX reward functions for Yahtzee meta-learning tasks.

Each function computes a scalar reward given score-step information.
All are jit-compatible. Dispatched via compute_reward() using jax.lax.switch.
"""
import jax
import jax.numpy as jnp

from src.env.constants import YAHTZEE

TASK_NAMES = ["MaxScore", "ThresholdBeater", "UpperBonus", "YahtzeeHunter", "Conservative"]


def reward_max_score(*, prev_total, new_total, category, score_gained,
                     done, final_score, yahtzee_bonus_delta, upper_crossed_63,
                     **kwargs) -> jnp.float32:
    return (new_total - prev_total) / 50.0


def reward_threshold_beater(*, prev_total, new_total, category, score_gained,
                            done, final_score, yahtzee_bonus_delta,
                            upper_crossed_63, threshold=250,
                            **kwargs) -> jnp.float32:
    return jnp.where(done & (final_score >= threshold), 1.0, 0.0)


def reward_upper_bonus(*, prev_total, new_total, category, score_gained,
                       done, final_score, yahtzee_bonus_delta, upper_crossed_63,
                       **kwargs) -> jnp.float32:
    is_upper = category < 6
    base = jnp.where(is_upper, score_gained / 3.0, 0.0)
    bonus = jnp.where(is_upper & upper_crossed_63, 35.0, 0.0)
    return base + bonus


def reward_yahtzee_hunter(*, prev_total, new_total, category, score_gained,
                          done, final_score, yahtzee_bonus_delta,
                          upper_crossed_63, **kwargs) -> jnp.float32:
    yahtzee_reward = jnp.where((category == YAHTZEE) & (score_gained > 0), 50.0, 0.0)
    return yahtzee_reward + yahtzee_bonus_delta


def reward_conservative(*, prev_total, new_total, category, score_gained,
                        done, final_score, yahtzee_bonus_delta, upper_crossed_63,
                        **kwargs) -> jnp.float32:
    return jnp.where(score_gained > 0, 1.0, -1.0)


def compute_reward(task_id, *, prev_total, new_total, category, score_gained,
                   done, final_score, yahtzee_bonus_delta, upper_crossed_63,
                   threshold=250) -> jnp.float32:
    """Dispatch reward computation by task_id via jax.lax.switch."""
    prev_total = jnp.asarray(prev_total, dtype=jnp.float32)
    new_total = jnp.asarray(new_total, dtype=jnp.float32)
    category = jnp.asarray(category, dtype=jnp.int32)
    score_gained = jnp.asarray(score_gained, dtype=jnp.float32)
    done = jnp.asarray(done, dtype=jnp.bool_)
    final_score = jnp.asarray(final_score, dtype=jnp.float32)
    yahtzee_bonus_delta = jnp.asarray(yahtzee_bonus_delta, dtype=jnp.float32)
    upper_crossed_63 = jnp.asarray(upper_crossed_63, dtype=jnp.bool_)
    threshold = jnp.asarray(threshold, dtype=jnp.float32)

    fns = [
        lambda args: reward_max_score(
            prev_total=args[0], new_total=args[1], category=args[2],
            score_gained=args[3], done=args[4], final_score=args[5],
            yahtzee_bonus_delta=args[6], upper_crossed_63=args[7]),
        lambda args: reward_threshold_beater(
            prev_total=args[0], new_total=args[1], category=args[2],
            score_gained=args[3], done=args[4], final_score=args[5],
            yahtzee_bonus_delta=args[6], upper_crossed_63=args[7],
            threshold=args[8]),
        lambda args: reward_upper_bonus(
            prev_total=args[0], new_total=args[1], category=args[2],
            score_gained=args[3], done=args[4], final_score=args[5],
            yahtzee_bonus_delta=args[6], upper_crossed_63=args[7]),
        lambda args: reward_yahtzee_hunter(
            prev_total=args[0], new_total=args[1], category=args[2],
            score_gained=args[3], done=args[4], final_score=args[5],
            yahtzee_bonus_delta=args[6], upper_crossed_63=args[7]),
        lambda args: reward_conservative(
            prev_total=args[0], new_total=args[1], category=args[2],
            score_gained=args[3], done=args[4], final_score=args[5],
            yahtzee_bonus_delta=args[6], upper_crossed_63=args[7]),
    ]

    args = (prev_total, new_total, category, score_gained, done,
            final_score, yahtzee_bonus_delta, upper_crossed_63, threshold)
    return jax.lax.switch(task_id, fns, args)
