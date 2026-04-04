"""Tests for pure-functional JAX Yahtzee environment."""
import jax
import jax.numpy as jnp
import pytest
from src.env.yahtzee_env import EnvState, env_reset, env_step, make_obs, get_action_mask
from src.env.constants import (
    N_CATEGORIES, N_DICE, PHASE_ROLL, PHASE_SCORE, MAX_REROLLS,
)


N_COLS = 6
OBS_DIM = 7 + 26 * N_COLS  # 163


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

    def test_dice_in_obs(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        obs = make_obs(state, N_COLS)
        assert jnp.array_equal(obs[:5], state.dice.astype(jnp.float32))


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

    def test_keep_all_transitions_to_score(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        new_state, obs, reward, done, info = env_step(state, jnp.int32(31), jnp.int32(0), N_COLS)
        assert int(new_state.phase) == PHASE_SCORE

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
        state, _, _, _, _ = env_step(state, jnp.int32(31), jnp.int32(0), N_COLS)
        assert int(state.phase) == PHASE_SCORE
        new_state, obs, reward, done, info = env_step(state, jnp.int32(0), jnp.int32(0), N_COLS)
        assert bool(new_state.filled_mask[0, 0])
        assert int(new_state.phase) == PHASE_ROLL

    def test_not_done_until_all_filled(self):
        rng = jax.random.PRNGKey(0)
        state, _ = env_reset(rng, N_COLS)
        state, _, _, _, _ = env_step(state, jnp.int32(31), jnp.int32(0), N_COLS)
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


class TestFullEpisode:
    def test_play_full_episode_terminates(self):
        rng = jax.random.PRNGKey(42)
        state, _ = env_reset(rng, N_COLS)
        done = False
        steps = 0
        max_steps = 300

        while not done and steps < max_steps:
            if int(state.phase) == PHASE_ROLL:
                action = jnp.int32(31)
            else:
                mask = get_action_mask(state, N_COLS)
                action = jnp.int32(jnp.argmax(mask))
            state, obs, reward, done, info = env_step(state, action, jnp.int32(0), N_COLS)
            done = bool(done)
            steps += 1

        assert done, f"Episode did not terminate within {max_steps} steps"
        assert steps <= max_steps
        assert jnp.all(state.filled_mask)
