"""Pure JAX reward functions for Yahtzee meta-learning tasks.

Each function computes a scalar reward given score-step information.
All are jit-compatible. Dispatched via compute_reward() using jax.lax.switch.
"""
import jax
import jax.numpy as jnp

from src.env.constants import YAHTZEE

TASK_NAMES = ["MaxScore", "ThresholdBeater", "UpperBonus", "YahtzeeHunter",
              "Conservative", "ExpectedValue"]


# Approximate expected score per category with decent play (2 rerolls).
# Upper section: 3-of-face is the upper bonus target (63 = 3+6+9+12+15+18).
# Lower section: expected value = max_score * P(hitting it with optimal rerolls).
# Used to shape rewards so rare/hard placements yield proportionally higher signal.
_CATEGORY_EXPECTED = jnp.array([
    2.1,   # Ones    (~3 target, often under)
    4.2,   # Twos
    6.3,   # Threes
    8.4,   # Fours
    10.5,  # Fives
    12.6,  # Sixes
    12.0,  # Three of a Kind  (sum of dice when hit, ~40% chance)
    5.0,   # Four of a Kind   (sum of dice when hit, ~15% chance)
    8.8,   # Full House       (25 * ~0.35)
    10.6,  # Small Straight   (30 * ~0.35)
    8.0,   # Large Straight   (40 * ~0.20)
    2.3,   # Yahtzee          (50 * ~0.046)
    20.0,  # Chance           (average dice sum ~17.5, decent aim ~20)
], dtype=jnp.float32)


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


def reward_expected_value(*, prev_total, new_total, category, score_gained,
                         done, final_score, yahtzee_bonus_delta,
                         upper_crossed_63, **kwargs) -> jnp.float32:
    """Reward shaped by how much better/worse than expected the placement is.

    reward = (score_gained - expected) / max(expected, 1)

    Cross-outs give ~ -1.0.  Hitting a Yahtzee gives ~ +20.
    Categories that are hard to score well yield larger magnitude rewards,
    giving the policy richer gradient signal than flat score/50.
    """
    expected = _CATEGORY_EXPECTED[category]
    return (score_gained - expected) / jnp.maximum(expected, 1.0)


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
        lambda args: reward_expected_value(
            prev_total=args[0], new_total=args[1], category=args[2],
            score_gained=args[3], done=args[4], final_score=args[5],
            yahtzee_bonus_delta=args[6], upper_crossed_63=args[7]),
    ]

    args = (prev_total, new_total, category, score_gained, done,
            final_score, yahtzee_bonus_delta, upper_crossed_63, threshold)
    return jax.lax.switch(task_id, fns, args)
