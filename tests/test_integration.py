"""Integration test: meta-steps, checkpoint save/load, eval trajectory schema."""
import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.agents.actor_critic import ActorCritic
from src.meta.maml import FOMAML
from src.training.evaluator import Evaluator
from src.tasks.reward_tasks import TASK_NAMES
from src.env.yahtzee_env import env_reset, env_step, get_action_mask
from src.env.constants import PHASE_ROLL, PHASE_SCORE, N_DICE


SMALL_CONFIG = {
    "env": {"n_columns": 3, "seed": 42, "max_steps_per_episode": 100},
    "agent": {"hidden_dim": 32, "n_layers": 1},
    "ppo": {"clip_epsilon": 0.2, "entropy_coef": 0.01, "value_loss_coef": 0.5, "gae_lambda": 0.95},
    "meta": {"inner_lr": 0.001, "outer_lr": 0.0003, "n_inner_steps": 2,
             "n_tasks_per_batch": 2, "n_meta_steps": 10, "n_parallel_envs": 2},
    "tasks": {"threshold_beater_score": 250},
    "training": {"checkpoint_every": 5, "checkpoint_dir": ""},
    "clearml": {"project_name": "test", "task_name": "test"},
}

SMALL_A2C_CONFIG = {
    "env": {"n_columns": 3, "seed": 42, "max_steps_per_episode": 100},
    "agent": {"hidden_dim": 32, "n_layers": 1},
    "a2c": {"entropy_coef": 0.01, "value_loss_coef": 0.5, "gae_lambda": 0.0},
    "meta": {"inner_lr": 0.001, "outer_lr": 0.0003, "n_inner_steps": 1,
             "n_tasks_per_batch": 2, "n_meta_steps": 10, "n_parallel_envs": 2},
    "tasks": {"threshold_beater_score": 250},
    "training": {"checkpoint_every": 5, "checkpoint_dir": ""},
    "clearml": {"project_name": "test", "task_name": "test"},
}


class TestMetaTraining:
    def test_10_meta_steps(self):
        config = SMALL_CONFIG
        model = ActorCritic(hidden_dim=32, n_layers=1, n_columns=3)
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)
        obs_dim = 7 + 26 * 3
        params = model.init(init_rng, jnp.zeros(obs_dim), jnp.int32(0))
        fomaml = FOMAML(model, config)
        opt_state = fomaml.init_optimizer(params)
        losses = []
        for step in range(10):
            result = fomaml.meta_update(params, opt_state, rng)
            params, opt_state, rng, meta_loss, task_losses, skipped = result
            if not skipped:
                losses.append(meta_loss)
        assert len(losses) > 0
        assert all(np.isfinite(l) for l in losses)

    def test_10_meta_steps_a2c(self):
        config = SMALL_A2C_CONFIG
        model = ActorCritic(hidden_dim=32, n_layers=1, n_columns=3)
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)
        obs_dim = 7 + 26 * 3
        params = model.init(init_rng, jnp.zeros(obs_dim), jnp.int32(0))
        fomaml = FOMAML(model, config)
        opt_state = fomaml.init_optimizer(params)
        losses = []
        for step in range(10):
            result = fomaml.meta_update(params, opt_state, rng)
            params, opt_state, rng, meta_loss, task_losses, skipped = result
            if not skipped:
                losses.append(meta_loss)
        assert len(losses) > 0
        assert all(np.isfinite(l) for l in losses)


class TestCheckpoint:
    def test_save_and_load_params(self, tmp_path):
        """Verify params can be saved and loaded back identically."""
        model = ActorCritic(hidden_dim=32, n_layers=1, n_columns=3)
        rng = jax.random.PRNGKey(42)
        obs_dim = 7 + 26 * 3
        params = model.init(rng, jnp.zeros(obs_dim), jnp.int32(0))

        # Save
        flat = {}
        for k, v in jax.tree_util.tree_leaves_with_path(jax.device_get(params)):
            key = "/".join(str(p) for p in k)
            flat[key] = np.asarray(v)
        np.savez(os.path.join(str(tmp_path), "params.npz"), **flat)

        # Load
        loaded = dict(np.load(os.path.join(str(tmp_path), "params.npz"), allow_pickle=True))
        flat_loaded = [jnp.array(loaded["/".join(str(p) for p in k)])
                       for k, _ in jax.tree_util.tree_leaves_with_path(params)]
        params_loaded = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(params), flat_loaded)

        # Verify
        for orig, load in zip(jax.tree.leaves(params), jax.tree.leaves(params_loaded)):
            assert jnp.array_equal(orig, load)


class TestEvaluation:
    def test_eval_trajectory_schema(self):
        config = SMALL_CONFIG
        model = ActorCritic(hidden_dim=32, n_layers=1, n_columns=3)
        rng = jax.random.PRNGKey(42)
        obs_dim = 7 + 26 * 3
        params = model.init(rng, jnp.zeros(obs_dim), jnp.int32(0))
        evaluator = Evaluator(config)
        df_steps, df_episodes = evaluator.evaluate(params, meta_step=0, n_episodes=2)

        # Step schema
        assert "episode_id" in df_steps.columns
        assert "phase" in df_steps.columns
        assert "value_estimate" in df_steps.columns
        assert "action_entropy" in df_steps.columns

        # Episode schema
        assert "final_score" in df_episodes.columns
        assert "strategy" in df_episodes.columns
        assert "n_turns" in df_episodes.columns
        assert "beat_threshold" in df_episodes.columns

        # Sanity checks
        assert all(df_steps["phase"].isin(["roll", "score"]))
        assert all(df_episodes["final_score"] >= 0)
        for d in range(1, N_DICE + 1):
            col = f"dice_{d}"
            if col in df_steps.columns:
                vals = df_steps[col].dropna()
                assert all((vals >= 1) & (vals <= 6))


class TestFullEpisodeViaEnv:
    def test_complete_game_n3(self):
        n_cols = 3
        rng = jax.random.PRNGKey(0)
        state, obs = env_reset(rng, n_cols)
        done = False
        steps = 0
        while not done and steps < 500:
            if int(state.phase) == PHASE_ROLL:
                action = jnp.int32(31)
            else:
                mask = get_action_mask(state, n_cols)
                action = jnp.int32(jnp.argmax(mask))
            state, obs, reward, done_flag, info = env_step(state, action, jnp.int32(0), n_cols)
            done = bool(done_flag)
            steps += 1
        assert done
        assert jnp.all(state.filled_mask)
