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
    YAHTZEE, PHASE_ROLL, PHASE_SCORE, MAX_CATEGORY_SCORES,
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
    result = jnp.where(keep_mask, dice, new_rolls)
    return jnp.sort(result)  # Paper: dice always sorted after roll


def env_reset(rng_key, n_columns):
    rng_key, dice_rng = jax.random.split(rng_key)
    dice = jnp.sort(jax.random.randint(dice_rng, (N_DICE,), 1, N_SIDES + 1))
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
    sorted_dice = jnp.sort(state.dice)
    dice_onehot = jax.nn.one_hot(sorted_dice - 1, N_SIDES).flatten()  # (30,)
    bin_counts = jnp.zeros(N_SIDES, dtype=jnp.float32).at[sorted_dice - 1].add(1.0)  # (6,)
    rolls_onehot = jax.nn.one_hot(state.rerolls, MAX_REROLLS + 1)  # (3,)

    all_scores = compute_all_scores(sorted_dice)
    potential_normalized = all_scores.astype(jnp.float32) / MAX_CATEGORY_SCORES  # (13,)
    all_dice_same = jnp.all(sorted_dice == sorted_dice[0])
    yahtzee_filled = jnp.any(state.scores[YAHTZEE, :] == 50)
    joker = jnp.array([all_dice_same & yahtzee_filled], dtype=jnp.float32)  # (1,)

    phase = jnp.array([state.phase], dtype=jnp.float32)  # (1,)
    has_yahtzee = jnp.array([yahtzee_filled], dtype=jnp.float32)  # (1,)

    upper_sums = jnp.sum(state.scores[:6, :], axis=0).astype(jnp.float32)
    upper_progress = jnp.minimum(upper_sums / UPPER_BONUS_THRESHOLD, 1.0)  # (n_cols,)

    upper_with_score = upper_sums + all_scores[:6, None]  # (6, n_cols) broadcast
    lockin = (upper_with_score >= UPPER_BONUS_THRESHOLD).astype(jnp.float32)
    lockin = lockin * (~state.filled_mask[:6, :]).astype(jnp.float32)
    lockin_flat = lockin.flatten()  # (6*n_cols,)

    n_filled = jnp.sum(state.filled_mask).astype(jnp.float32)
    total_slots = N_CATEGORIES * n_columns
    game_progress = jnp.array([n_filled / total_slots])  # (1,)

    return jnp.concatenate([
        dice_onehot,                                           # 30
        bin_counts,                                            # 6
        rolls_onehot,                                          # 3
        potential_normalized,                                  # 13
        joker,                                                 # 1
        phase,                                                 # 1
        has_yahtzee,                                           # 1
        state.filled_mask.flatten().astype(jnp.float32),       # 13 * n_cols
        state.scores.flatten().astype(jnp.float32),            # 13 * n_cols
        jnp.array([state.yahtzee_bonus], dtype=jnp.float32),   # 1
        upper_progress,                                        # n_cols
        lockin_flat,                                           # 6 * n_cols
        game_progress,                                         # 1
    ])


def make_obs_paper(state, n_columns):
    """Paper-exact 76-dim observation (Pape 2025, single-column).

    Feature order matches full_game_a2c.json phi_features config:
    dice_onehot(30) + dice_counts(6) + rolls_used(3) + phase(1) +
    has_earned_yahtzee(1) + available_categories(13) +
    percent_progress_towards_bonus(1) + potential_scoring_opportunities(14) +
    game_progress(1) + will_receive_bonus_if_chosen(6) = 76
    """
    sorted_dice = jnp.sort(state.dice)

    dice_onehot = jax.nn.one_hot(sorted_dice - 1, N_SIDES).flatten()  # (30,)
    dice_counts = jnp.zeros(N_SIDES, dtype=jnp.float32).at[sorted_dice - 1].add(1.0)  # (6,)
    rolls_onehot = jax.nn.one_hot(state.rerolls, MAX_REROLLS + 1)  # (3,)
    phase_f = jnp.array([state.phase], dtype=jnp.float32)  # (1,)

    yahtzee_filled = jnp.any(state.scores[YAHTZEE, :] == 50)
    has_yahtzee = jnp.array([yahtzee_filled], dtype=jnp.float32)  # (1,)

    # available_categories: 1=available, 0=filled (INVERTED from filled_mask)
    available = (~state.filled_mask).astype(jnp.float32).flatten()  # (13*n_cols,)
    # For 1-col this is (13,). For multi-col, flatten all columns.

    # percent_progress_towards_bonus: NOT clamped (can exceed 1.0)
    upper_sum = jnp.sum(state.scores[:6, :]).astype(jnp.float32)
    bonus_progress = jnp.array([upper_sum / UPPER_BONUS_THRESHOLD])  # (1,)

    # potential_scoring_opportunities: masked by available categories + joker
    all_scores = compute_all_scores(sorted_dice)
    # Paper: scores for filled categories are 0. For multi-col, a category
    # is "available" if ANY column has it unfilled.
    avail_any_col = jnp.any(~state.filled_mask, axis=1).astype(jnp.float32)  # (13,)
    masked_scores = all_scores.astype(jnp.float32) * avail_any_col
    potential_normalized = masked_scores / MAX_CATEGORY_SCORES  # (13,)
    all_dice_same = jnp.all(sorted_dice == sorted_dice[0])
    joker = jnp.array([all_dice_same & yahtzee_filled], dtype=jnp.float32)  # (1,)

    # game_progress: 1 - (available/total)
    n_available = jnp.sum(~state.filled_mask).astype(jnp.float32)
    total_slots = N_CATEGORIES * n_columns
    game_progress = jnp.array([1.0 - n_available / total_slots])  # (1,)

    # will_receive_bonus_if_chosen: per upper category per column
    upper_sums_per_col = jnp.sum(state.scores[:6, :], axis=0).astype(jnp.float32)
    upper_with_score = upper_sums_per_col + all_scores[:6, None]  # (6, n_cols)
    lockin = (upper_with_score >= UPPER_BONUS_THRESHOLD).astype(jnp.float32)
    lockin = lockin * (~state.filled_mask[:6, :]).astype(jnp.float32)
    lockin_flat = lockin.flatten()  # (6*n_cols,)

    return jnp.concatenate([
        dice_onehot,           # 30
        dice_counts,           # 6
        rolls_onehot,          # 3
        phase_f,               # 1
        has_yahtzee,           # 1
        available,             # 13 * n_cols
        bonus_progress,        # 1
        potential_normalized,  # 13
        joker,                 # 1
        game_progress,         # 1
        lockin_flat,           # 6 * n_cols
    ])


def paper_obs_dim(n_columns):
    """Paper-exact observation dimension: 57 + 19 * n_columns."""
    return 57 + 19 * n_columns


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


def get_action_mask_paper(state, n_columns):
    """Paper-compatible action mask: ALL unfilled categories are valid for scoring.

    Unlike get_action_mask which only allows categories with score > 0,
    this allows strategic cross-outs (scoring 0 in a category to save
    better ones for later). Matches the paper's MaskedSoftmax behavior.
    """
    max_actions = max(32, 13 * n_columns)

    roll_mask = jnp.concatenate([
        jnp.ones(32, dtype=jnp.bool_),
        jnp.zeros(max_actions - 32, dtype=jnp.bool_),
    ])

    score_mask = (~state.filled_mask).flatten()
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


def _roll_step_paper(state, action, n_columns):
    """Paper-compatible roll step: no early score transition.

    Transition to score phase ONLY when all rerolls are used (rerolls <= 0).
    Action 31 (keep all) does NOT skip to score — it just keeps all dice.
    This ensures every turn is exactly 3 steps: roll, roll, score.
    """
    keep_mask = jnp.array([(action >> i) & 1 for i in range(N_DICE)], dtype=jnp.bool_)
    rng_key, dice_rng = jax.random.split(state.rng_key)
    new_dice = _roll_dice(dice_rng, state.dice, keep_mask)
    new_rerolls = state.rerolls - 1
    go_to_score = new_rerolls <= 0  # NO action==31 shortcut
    new_phase = jnp.where(go_to_score, PHASE_SCORE, PHASE_ROLL)
    new_state = state._replace(
        dice=new_dice, rerolls=new_rerolls, phase=new_phase,
        rng_key=rng_key, step_count=state.step_count + 1,
    )
    return new_state, jnp.float32(0.0), jnp.bool_(False)


def _compute_true_total(scores, yahtzee_bonus):
    """Compute total score including upper bonuses and yahtzee bonuses."""
    raw = jnp.sum(scores)
    upper_per_col = jnp.sum(scores[:6, :], axis=0)
    upper_bonuses = jnp.sum(jnp.where(upper_per_col >= UPPER_BONUS_THRESHOLD,
                                       UPPER_BONUS_VALUE, 0))
    return raw + upper_bonuses + yahtzee_bonus


def _score_step(state, action, task_id, n_columns, threshold):
    category = action // n_columns
    column = action % n_columns

    score_val = score_category(state.dice, category)
    prev_total = _compute_true_total(state.scores, state.yahtzee_bonus)
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

    new_total = _compute_true_total(new_scores, new_yahtzee_bonus)

    done = jnp.all(new_filled)

    rng_key, dice_rng, next_rng = jax.random.split(state.rng_key, 3)
    next_dice = jnp.sort(jax.random.randint(dice_rng, (N_DICE,), 1, N_SIDES + 1))

    reward = compute_reward(
        task_id=task_id,
        prev_total=prev_total, new_total=new_total,
        category=category, score_gained=score_val,
        done=done, final_score=new_total,
        yahtzee_bonus_delta=yahtzee_bonus_delta,
        upper_crossed_63=newly_crossed_63, threshold=threshold,
    )

    new_state = state._replace(
        dice=next_dice, rerolls=jnp.int32(MAX_REROLLS),
        filled_mask=new_filled, scores=new_scores,
        yahtzee_bonus=new_yahtzee_bonus,
        phase=jnp.where(done, PHASE_SCORE, PHASE_ROLL),
        rng_key=next_rng, step_count=state.step_count + 1,
    )
    return new_state, reward, done


def env_step(state, action, task_id, n_columns, threshold=250):
    new_state, reward, done = jax.lax.cond(
        state.phase == PHASE_ROLL,
        lambda s, a, t: _roll_step(s, a, n_columns),
        lambda s, a, t: _score_step(s, a, t, n_columns, threshold),
        state, action, task_id,
    )
    obs = make_obs(new_state, n_columns)
    info = {"phase": new_state.phase}
    return new_state, obs, reward, done, info


def env_step_paper(state, action, task_id, n_columns, threshold=250):
    """Paper-compatible env step: no early score transition on keep-all.

    Every turn takes exactly 3 steps (roll, roll, score) = 39 steps per game.
    """
    new_state, reward, done = jax.lax.cond(
        state.phase == PHASE_ROLL,
        lambda s, a, t: _roll_step_paper(s, a, n_columns),
        lambda s, a, t: _score_step(s, a, t, n_columns, threshold),
        state, action, task_id,
    )
    obs = make_obs_paper(new_state, n_columns)
    info = {"phase": new_state.phase}
    return new_state, obs, reward, done, info


