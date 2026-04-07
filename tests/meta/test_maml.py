"""Tests for FOMAML inner loop, episode collection, and meta-update."""
import jax
import jax.numpy as jnp
import pytest
from src.agents.actor_critic import ActorCritic
from src.agents.ppo import ppo_loss
from src.meta.inner_loop import collect_episodes, inner_update_and_query_grad, prepare_ppo_data

N_COLS = 6
OBS_DIM = 7 + 26 * N_COLS
MAX_ACTIONS = max(32, 13 * N_COLS)


@pytest.fixture
def model_and_params():
    model = ActorCritic(hidden_dim=32, n_layers=1, n_columns=N_COLS)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
    return model, params


class TestCollectEpisodes:
    def test_returns_trajectory_tuple(self, model_and_params):
        model, params = model_and_params
        rng = jax.random.PRNGKey(42)
        traj = collect_episodes(params, model, rng, jnp.int32(0),
                                n_parallel_envs=2, n_columns=N_COLS, max_steps=50)
        obs, actions, log_probs, rewards, dones, values, phases, masks = traj
        assert obs.shape == (50, 2, OBS_DIM)
        assert actions.shape == (50, 2)
        assert rewards.shape == (50, 2)
        assert dones.shape == (50, 2)
        assert values.shape == (50, 2)
        assert phases.shape == (50, 2)
        assert masks.shape == (50, 2, MAX_ACTIONS)

    def test_obs_are_finite(self, model_and_params):
        model, params = model_and_params
        rng = jax.random.PRNGKey(42)
        traj = collect_episodes(params, model, rng, jnp.int32(0),
                                n_parallel_envs=2, n_columns=N_COLS, max_steps=50)
        assert jnp.all(jnp.isfinite(traj[0]))

    def test_different_seeds_different_results(self, model_and_params):
        model, params = model_and_params
        t1 = collect_episodes(params, model, jax.random.PRNGKey(0),
                              jnp.int32(0), 2, N_COLS, 50)
        t2 = collect_episodes(params, model, jax.random.PRNGKey(1),
                              jnp.int32(0), 2, N_COLS, 50)
        assert not jnp.array_equal(t1[0], t2[0])


class TestPreparePPOData:
    def test_output_shapes(self, model_and_params):
        model, params = model_and_params
        rng = jax.random.PRNGKey(42)
        traj = collect_episodes(params, model, rng, jnp.int32(0),
                                n_parallel_envs=2, n_columns=N_COLS, max_steps=50)
        data = prepare_ppo_data(traj)
        obs, actions, phases, masks, old_log_probs, advantages, returns = data
        n = obs.shape[0]
        assert actions.shape == (n,)
        assert phases.shape == (n,)
        assert masks.shape == (n, MAX_ACTIONS)
        assert old_log_probs.shape == (n,)
        assert advantages.shape == (n,)
        assert returns.shape == (n,)

    def test_advantages_normalized(self, model_and_params):
        model, params = model_and_params
        rng = jax.random.PRNGKey(42)
        traj = collect_episodes(params, model, rng, jnp.int32(0),
                                n_parallel_envs=4, n_columns=N_COLS, max_steps=50)
        data = prepare_ppo_data(traj)
        advantages = data[5]
        assert jnp.abs(jnp.mean(advantages)) < 0.5


class TestInnerUpdateAndQueryGrad:
    def test_returns_grads_and_losses(self, model_and_params):
        model, params = model_and_params
        rng = jax.random.PRNGKey(42)
        rng1, rng2 = jax.random.split(rng)
        support_traj = collect_episodes(params, model, rng1, jnp.int32(0), 2, N_COLS, 50)
        query_traj = collect_episodes(params, model, rng2, jnp.int32(0), 2, N_COLS, 50)
        support_data = prepare_ppo_data(support_traj)
        query_data = prepare_ppo_data(query_traj)
        config = {"inner_lr": 0.001, "n_inner_steps": 3}
        grads, query_loss, inner_losses = inner_update_and_query_grad(
            params, model, ppo_loss, support_data, query_data, config)
        grad_leaves = jax.tree.leaves(grads)
        param_leaves = jax.tree.leaves(params)
        assert len(grad_leaves) == len(param_leaves)
        for g, p in zip(grad_leaves, param_leaves):
            assert g.shape == p.shape
        assert query_loss.shape == ()
        assert jnp.isfinite(query_loss)
        assert inner_losses.shape == (3,)

    def test_inner_update_produces_nonzero_grads(self, model_and_params):
        model, params = model_and_params
        rng = jax.random.PRNGKey(42)
        rng1, rng2 = jax.random.split(rng)
        support_traj = collect_episodes(params, model, rng1, jnp.int32(0), 2, N_COLS, 50)
        query_traj = collect_episodes(params, model, rng2, jnp.int32(0), 2, N_COLS, 50)
        support_data = prepare_ppo_data(support_traj)
        query_data = prepare_ppo_data(query_traj)
        config = {"inner_lr": 0.01, "n_inner_steps": 5}
        grads, _, _ = inner_update_and_query_grad(
            params, model, ppo_loss, support_data, query_data, config)
        has_nonzero = any(jnp.any(g != 0) for g in jax.tree.leaves(grads))
        assert has_nonzero
