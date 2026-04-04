"""Pure-functional JAX Yahtzee environment.

All functions are pure (no side effects) and jit-compatible.
State is a NamedTuple pytree. Use jax.vmap for batched environments.
"""
import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple

from src.env.constants import (
    N_DICE, N_SIDES, N_CATEGORIES, MAX_REROLLS,
    UPPER_BONUS_THRESHOLD, UPPER_BONUS_VALUE, YAHTZEE_BONUS_VALUE,
    YAHTZEE, PHASE_ROLL, PHASE_SCORE,
)
from src.env.scoring import score_category, compute_all_scores
from src.tasks.reward_tasks import compute_reward


class EnvState(NamedTuple):
    dice: jnp.ndarray           # (5,) int32
    rerolls: jnp.int32          # 0-2
    filled_mask: jnp.ndarray    # (13, n_cols) bool
    scores: jnp.ndarray         # (13, n_cols) int32
    yahtzee_bonus: jnp.int32    # cumulative
    phase: jnp.int32            # 0=roll, 1=score
    rng_key: jnp.ndarray        # PRNGKey
    step_count: jnp.int32       # episode length


def _roll_dice(rng_key, dice, keep_mask):
    new_rolls = jax.random.randint(rng_key, (N_DICE,), 1, N_SIDES + 1)
    return jnp.where(keep_mask, dice, new_rolls)


def env_reset(rng_key, n_columns):
    rng_key, dice_rng = jax.random.split(rng_key)
    dice = jax.random.randint(dice_rng, (N_DICE,), 1, N_SIDES + 1)
    state = EnvState(
        dice=dice,
        rerolls=jnp.int32(MAX_REROLLS),
        filled_mask=jnp.zeros((N_CATEGORIES, n_columns), dtype=jnp.bool_),
        scores=jnp.zeros((N_CATEGORIES, n_columns), dtype=jnp.int32),
        yahtzee_bonus=jnp.int32(0),
        phase=jnp.int32(PHASE_ROLL),
        rng_key=rng_key,
        step_count=jnp.int32(0),
    )
    return state, make_obs(state, n_columns)


def make_obs(state, n_columns):
    return jnp.concatenate([
        state.dice.astype(jnp.float32),
        jnp.array([state.rerolls], dtype=jnp.float32),
        state.filled_mask.flatten().astype(jnp.float32),
        state.scores.flatten().astype(jnp.float32),
        jnp.array([state.yahtzee_bonus], dtype=jnp.float32),
    ])


def get_action_mask(state, n_columns):
    max_actions = max(32, 13 * n_columns)

    roll_mask = jnp.concatenate([
        jnp.ones(32, dtype=jnp.bool_),
        jnp.zeros(max_actions - 32, dtype=jnp.bool_),
    ])

    all_scores = compute_all_scores(state.dice)
    can_score = (~state.filled_mask) & (all_scores[:, None] > 0)
    any_positive = jnp.any(can_score)
    score_mask_2d = jnp.where(any_positive, can_score, ~state.filled_mask)
    score_mask = score_mask_2d.flatten()

    score_mask_padded = jnp.zeros(max_actions, dtype=jnp.bool_)
    score_mask_padded = score_mask_padded.at[:13 * n_columns].set(score_mask)

    return jnp.where(state.phase == PHASE_ROLL, roll_mask, score_mask_padded)


def _roll_step(state, action, n_columns):
    keep_mask = jnp.array([(action >> i) & 1 for i in range(N_DICE)], dtype=jnp.bool_)
    rng_key, dice_rng = jax.random.split(state.rng_key)
    new_dice = _roll_dice(dice_rng, state.dice, keep_mask)
    new_rerolls = state.rerolls - 1
    go_to_score = (new_rerolls <= 0) | (action == 31)
    new_phase = jnp.where(go_to_score, PHASE_SCORE, PHASE_ROLL)
    new_state = state._replace(
        dice=new_dice, rerolls=new_rerolls, phase=new_phase,
        rng_key=rng_key, step_count=state.step_count + 1,
    )
    return new_state, jnp.float32(0.0), jnp.bool_(False)


def _score_step(state, action, task_id, n_columns):
    category = action // n_columns
    column = action % n_columns

    score_val = score_category(state.dice, category)
    prev_total = jnp.sum(state.scores)
    prev_upper_col = jnp.sum(state.scores[:6, column])

    new_scores = state.scores.at[category, column].set(score_val)
    new_filled = state.filled_mask.at[category, column].set(True)

    new_upper_col = jnp.sum(new_scores[:6, column])
    newly_crossed_63 = (prev_upper_col < UPPER_BONUS_THRESHOLD) & \
                       (new_upper_col >= UPPER_BONUS_THRESHOLD)

    all_dice_same = jnp.all(state.dice == state.dice[0])
    all_yahtzee_slots_filled = jnp.all(state.scores[YAHTZEE, :] == 50)
    yahtzee_bonus_triggered = all_dice_same & all_yahtzee_slots_filled
    yahtzee_bonus_delta = jnp.where(yahtzee_bonus_triggered, YAHTZEE_BONUS_VALUE, 0)
    new_yahtzee_bonus = state.yahtzee_bonus + yahtzee_bonus_delta

    upper_bonus_delta = jnp.where(newly_crossed_63, UPPER_BONUS_VALUE, 0)
    new_total = jnp.sum(new_scores) + upper_bonus_delta + yahtzee_bonus_delta

    done = jnp.all(new_filled)

    rng_key, dice_rng, next_rng = jax.random.split(state.rng_key, 3)
    next_dice = jax.random.randint(dice_rng, (N_DICE,), 1, N_SIDES + 1)

    reward = compute_reward(
        task_id=task_id,
        prev_total=prev_total, new_total=new_total,
        category=category, score_gained=score_val,
        done=done, final_score=new_total,
        yahtzee_bonus_delta=yahtzee_bonus_delta,
        upper_crossed_63=newly_crossed_63, threshold=250,
    )

    new_state = state._replace(
        dice=next_dice, rerolls=jnp.int32(MAX_REROLLS),
        filled_mask=new_filled, scores=new_scores,
        yahtzee_bonus=new_yahtzee_bonus,
        phase=jnp.where(done, PHASE_SCORE, PHASE_ROLL),
        rng_key=next_rng, step_count=state.step_count + 1,
    )
    return new_state, reward, done


def env_step(state, action, task_id, n_columns):
    new_state, reward, done = jax.lax.cond(
        state.phase == PHASE_ROLL,
        lambda s, a, t: _roll_step(s, a, n_columns),
        lambda s, a, t: _score_step(s, a, t, n_columns),
        state, action, task_id,
    )
    obs = make_obs(new_state, n_columns)
    info = {"phase": new_state.phase}
    return new_state, obs, reward, done, info
