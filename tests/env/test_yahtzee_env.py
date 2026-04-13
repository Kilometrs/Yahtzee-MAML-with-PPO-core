"""Tests for pure-functional JAX Yahtzee environment."""
import jax
import jax.numpy as jnp
import pytest
from src.env.yahtzee_env import EnvState, env_reset, env_step, make_obs, get_action_mask
from src.env.constants import (
    N_CATEGORIES, N_DICE, N_SIDES, PHASE_ROLL, PHASE_SCORE, MAX_REROLLS,
    obs_dim,
)


N_COLS = 6
OBS_DIM = obs_dim(N_COLS)  # 255


class TestObsDim:
    def test_obs_dim_formula(self):
        assert obs_dim(1) == 57 + 33 * 1
        assert obs_dim(3) == 57 + 33 * 3
        assert obs_dim(6) == 57 + 33 * 6


class TestEnvReset:
    def test_obs_shape(self):
        rng = jax.random.PRNGKey(0)
        state, obs = env_reset(rng, N_COLS)
        assert obs.shape == (OBS_DIM,)

    def test_initial_phase_is_roll(self):
        rng = jax.random.PRNGKey(0)
        state, obs = env_reset(rng, N_COLS)
        assert int(state.phase) == PHASE_ROLL

    def test_initial_rerolls(self):
        rng = jax.random.PRNGKey(0)
        state, obs = env_reset(rng, N_COLS)
        assert int(state.rerolls) == MAX_REROLLS

    def test_dice_in_range(self):
        rng = jax.random.PRNGKey(42)
        state, obs = env_reset(rng, N_COLS)
        assert jnp.all((state.dice >= 1) & (state.dice <= 6))

    def test_filled_mask_empty(self):
        rng = jax.random.PRNGKey(0)
        state, obs = env_reset(rng, N_COLS)
        assert state.filled_mask.shape == (N_CATEGORIES, N_COLS)
        assert not jnp.any(state.filled_mask)

    def test_scores_zero(self):
        rng = jax.random.PRNGKey(0)
        state, obs = env_reset(rng, N_COLS)
        assert state.scores.shape == (N_CATEGORIES, N_COLS)
        assert jnp.all(state.scores == 0)

    def test_is_jittable(self):
        rng = jax.random.PRNGKey(0)
        state, obs = jax.jit(env_reset, static_argnums=(1,))(rng, N_COLS)
        assert obs.shape == (OBS_DIM,)

    def test_vmappable(self):
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        states, obs_batch = jax.vmap(env_reset, in_axes=(0, None))(keys, N_COLS)
        assert obs_batch.shape == (4, OBS_DIM)
        assert states.dice.shape == (4, N_DICE)


class TestMakeObs:
    def test_shape(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        obs = make_obs(state, N_COLS)
        assert obs.shape == (OBS_DIM,)

    def test_obs_dtype_float32(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        obs = make_obs(state, N_COLS)
        assert obs.dtype == jnp.float32


class TestMakeObsFeatures:
    def test_dice_onehot_section(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        obs = make_obs(state, N_COLS)
        dice_oh = obs[:30].reshape(5, 6)
        assert jnp.allclose(jnp.sum(dice_oh, axis=1), 1.0)

    def test_bin_counts_section(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        obs = make_obs(state, N_COLS)
        assert float(jnp.sum(obs[30:36])) == 5.0

    def test_rolls_onehot_section(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        obs = make_obs(state, N_COLS)
        rolls = obs[36:39]
        assert float(jnp.sum(rolls)) == 1.0
        assert float(rolls[MAX_REROLLS]) == 1.0

    def test_potential_scores_section(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        obs = make_obs(state, N_COLS)
        potential = obs[39:52]
        assert potential.shape == (13,)
        assert jnp.all(potential >= 0.0)

    def test_game_progress_starts_zero(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        obs = make_obs(state, N_COLS)
        assert float(obs[-1]) == 0.0

    def test_upper_progress_starts_zero(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        obs = make_obs(state, N_COLS)
        offset = 55 + 26 * N_COLS + 1
        upper_progress = obs[offset:offset + N_COLS]
        assert jnp.allclose(upper_progress, 0.0)


class TestGetActionMask:
    def test_roll_phase_mask_shape(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        max_actions = max(32, 13 * N_COLS)
        mask = get_action_mask(state, N_COLS)
        assert mask.shape == (max_actions,)

    def test_roll_phase_all_valid(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        mask = get_action_mask(state, N_COLS)
        assert jnp.all(mask[:32])

    def test_roll_phase_padding_false(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        mask = get_action_mask(state, N_COLS)
        max_actions = max(32, 13 * N_COLS)
        if max_actions > 32:
            assert not jnp.any(mask[32:])


class TestEnvStep:
    def test_roll_step_decrements_rerolls(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        new_state, obs, reward, done, info = env_step(state, jnp.int32(0), jnp.int32(0), N_COLS)
        assert int(new_state.rerolls) == MAX_REROLLS - 1
        assert int(new_state.phase) == PHASE_ROLL

    def test_keep_all_does_not_transition_to_score(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        new_state, obs, reward, done, info = env_step(state, jnp.int32(31), jnp.int32(0), N_COLS)
        assert int(new_state.phase) == PHASE_ROLL
        assert int(new_state.rerolls) == MAX_REROLLS - 1

    def test_exhaust_rerolls_transitions_to_score(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        state, _, _, _, _ = env_step(state, jnp.int32(0), jnp.int32(0), N_COLS)
        assert int(state.rerolls) == 1
        state, _, _, _, _ = env_step(state, jnp.int32(0), jnp.int32(0), N_COLS)
        assert int(state.phase) == PHASE_SCORE

    def test_score_step_fills_slot(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        # exhaust rerolls to get to score phase
        state, _, _, _, _ = env_step(state, jnp.int32(0), jnp.int32(0), N_COLS)
        state, _, _, _, _ = env_step(state, jnp.int32(0), jnp.int32(0), N_COLS)
        assert int(state.phase) == PHASE_SCORE
        new_state, obs, reward, done, info = env_step(state, jnp.int32(0), jnp.int32(0), N_COLS)
        assert bool(new_state.filled_mask[0, 0])
        assert int(new_state.phase) == PHASE_ROLL

    def test_not_done_until_all_filled(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        # exhaust rerolls to get to score phase
        state, _, _, _, _ = env_step(state, jnp.int32(0), jnp.int32(0), N_COLS)
        state, _, _, _, _ = env_step(state, jnp.int32(0), jnp.int32(0), N_COLS)
        assert int(state.phase) == PHASE_SCORE
        _, _, _, done, _ = env_step(state, jnp.int32(0), jnp.int32(0), N_COLS)
        assert not bool(done)

    def test_reward_zero_during_roll(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        _, _, reward, _, _ = env_step(state, jnp.int32(0), jnp.int32(0), N_COLS)
        assert float(reward) == 0.0

    def test_is_jittable(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        step_jit = jax.jit(env_step, static_argnums=(3,))
        new_state, obs, reward, done, info = step_jit(state, jnp.int32(0), jnp.int32(0), N_COLS)
        assert obs.shape == (OBS_DIM,)

    def test_vmappable(self):
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        states, _ = jax.vmap(env_reset, in_axes=(0, None))(keys, N_COLS)
        actions = jnp.zeros(4, dtype=jnp.int32)
        step_fn = jax.vmap(env_step, in_axes=(0, 0, None, None))
        new_states, obs, rewards, dones, infos = step_fn(states, actions, jnp.int32(0), N_COLS)
        assert obs.shape == (4, OBS_DIM)


class TestDiceSorting:
    def test_reset_dice_sorted(self):
        rng = jax.random.PRNGKey(42)
        state, _ = env_reset(rng, N_COLS)
        assert jnp.array_equal(state.dice, jnp.sort(state.dice))

    def test_roll_step_dice_sorted(self):
        rng = jax.random.PRNGKey(42)
        state, _ = env_reset(rng, N_COLS)
        new_state, _, _, _, _ = env_step(state, jnp.int32(0), jnp.int32(0), N_COLS)
        assert jnp.array_equal(new_state.dice, jnp.sort(new_state.dice))

    def test_score_step_next_dice_sorted(self):
        rng = jax.random.PRNGKey(42)
        state, _ = env_reset(rng, N_COLS)
        state, _, _, _, _ = env_step(state, jnp.int32(0), jnp.int32(0), N_COLS)
        state, _, _, _, _ = env_step(state, jnp.int32(0), jnp.int32(0), N_COLS)
        assert int(state.phase) == PHASE_SCORE
        mask = get_action_mask(state, N_COLS)
        action = jnp.int32(jnp.argmax(mask))
        new_state, _, _, _, _ = env_step(state, action, jnp.int32(0), N_COLS)
        assert jnp.array_equal(new_state.dice, jnp.sort(new_state.dice))


class TestNoEarlyExit:
    def test_keep_all_does_not_skip_to_score(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        assert int(state.rerolls) == MAX_REROLLS
        new_state, _, _, _, _ = env_step(state, jnp.int32(31), jnp.int32(0), N_COLS)
        assert int(new_state.phase) == PHASE_ROLL
        assert int(new_state.rerolls) == MAX_REROLLS - 1

    def test_episode_exactly_39_steps_1col(self):
        n_cols = 1
        rng = jax.random.PRNGKey(42)
        state, _ = env_reset(rng, n_cols)
        done = False
        steps = 0
        while not done and steps < 100:
            if int(state.phase) == PHASE_ROLL:
                action = jnp.int32(0)
            else:
                mask = get_action_mask(state, n_cols)
                action = jnp.int32(jnp.argmax(mask))
            state, _, _, done_flag, _ = env_step(state, action, jnp.int32(0), n_cols)
            done = bool(done_flag)
            steps += 1
        assert done
        assert steps == 39


class TestStrategicCrossOuts:
    def test_all_unfilled_categories_valid(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        state, _, _, _, _ = env_step(state, jnp.int32(0), jnp.int32(0), N_COLS)
        state, _, _, _, _ = env_step(state, jnp.int32(0), jnp.int32(0), N_COLS)
        assert int(state.phase) == PHASE_SCORE
        mask = get_action_mask(state, N_COLS)
        assert jnp.sum(mask[:13 * N_COLS]) == 13 * N_COLS


class TestFullEpisode:
    def test_play_full_episode_terminates(self):
        rng = jax.random.PRNGKey(42)
        state, _ = env_reset(rng, N_COLS)
        done = False
        steps = 0
        max_steps = 500

        while not done and steps < max_steps:
            if int(state.phase) == PHASE_ROLL:
                action = jnp.int32(0)
            else:
                mask = get_action_mask(state, N_COLS)
                action = jnp.int32(jnp.argmax(mask))
            state, obs, reward, done, info = env_step(state, action, jnp.int32(0), N_COLS)
            done = bool(done)
            steps += 1

        assert done, f"Episode did not terminate within {max_steps} steps"
        assert steps <= max_steps
        assert jnp.all(state.filled_mask)
