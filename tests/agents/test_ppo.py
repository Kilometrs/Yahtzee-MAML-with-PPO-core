"""Tests for JAX GAE computation and PPO loss."""
import jax
import jax.numpy as jnp
import pytest
from src.agents.ppo import compute_gae, ppo_loss
from src.agents.actor_critic import ActorCritic
from src.env.constants import obs_dim

N_COLS = 6
OBS_DIM = obs_dim(N_COLS)
MAX_ACTIONS = max(32, 13 * N_COLS)


class TestComputeGAE:
    def test_output_shapes(self):
        rewards = jnp.array([1.0, 2.0, 3.0])
        values = jnp.array([0.5, 0.5, 0.5])
        dones = jnp.array([0.0, 0.0, 1.0])
        advantages, returns = compute_gae(rewards, values, dones)
        assert advantages.shape == (3,)
        assert returns.shape == (3,)

    def test_returns_equal_advantages_plus_values(self):
        rewards = jnp.array([1.0, 2.0, 3.0])
        values = jnp.array([0.5, 0.5, 0.5])
        dones = jnp.array([0.0, 0.0, 1.0])
        advantages, returns = compute_gae(rewards, values, dones)
        assert jnp.allclose(returns, advantages + values, atol=1e-5)

    def test_single_step_done(self):
        rewards = jnp.array([10.0])
        values = jnp.array([3.0])
        dones = jnp.array([1.0])
        advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        expected_adv = 10.0 - 3.0
        assert jnp.allclose(advantages[0], expected_adv, atol=1e-5)

    def test_two_steps_no_done(self):
        gamma, lam = 0.99, 0.95
        rewards = jnp.array([1.0, 2.0])
        values = jnp.array([0.5, 0.5])
        dones = jnp.array([0.0, 0.0])
        advantages, returns = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)
        assert jnp.allclose(advantages[1], 1.5, atol=1e-5)
        assert jnp.allclose(advantages[0], 2.40575, atol=1e-4)

    def test_is_jittable(self):
        rewards = jnp.array([1.0, 2.0])
        values = jnp.array([0.5, 0.5])
        dones = jnp.array([0.0, 1.0])
        jitted = jax.jit(compute_gae)
        advantages, returns = jitted(rewards, values, dones)
        assert advantages.shape == (2,)


class TestPPOLoss:
    @pytest.fixture
    def setup(self):
        model = ActorCritic(hidden_dim=32, n_layers=1, n_columns=N_COLS)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
        n_steps = 8
        obs = jax.random.normal(jax.random.PRNGKey(1), (n_steps, OBS_DIM))
        actions = jax.random.randint(jax.random.PRNGKey(2), (n_steps,), 0, 32)
        phases = jnp.zeros(n_steps, dtype=jnp.int32)
        masks = jnp.concatenate([
            jnp.ones((n_steps, 32), dtype=jnp.bool_),
            jnp.zeros((n_steps, MAX_ACTIONS - 32), dtype=jnp.bool_),
        ], axis=1)
        old_log_probs = -jnp.ones(n_steps)
        advantages = jnp.ones(n_steps)
        returns = jnp.ones(n_steps)
        return dict(model=model, params=params, obs=obs, actions=actions,
                    phases=phases, masks=masks, old_log_probs=old_log_probs,
                    advantages=advantages, returns=returns)

    def test_returns_scalar(self, setup):
        loss = ppo_loss(setup["params"], setup["model"], setup["obs"],
                        setup["actions"], setup["phases"], setup["masks"],
                        setup["old_log_probs"], setup["advantages"], setup["returns"])
        assert loss.shape == ()

    def test_loss_is_finite(self, setup):
        loss = ppo_loss(setup["params"], setup["model"], setup["obs"],
                        setup["actions"], setup["phases"], setup["masks"],
                        setup["old_log_probs"], setup["advantages"], setup["returns"])
        assert jnp.isfinite(loss)

    def test_differentiable(self, setup):
        def loss_fn(p):
            return ppo_loss(p, setup["model"], setup["obs"], setup["actions"],
                            setup["phases"], setup["masks"], setup["old_log_probs"],
                            setup["advantages"], setup["returns"])
        grads = jax.grad(loss_fn)(setup["params"])
        leaves = jax.tree.leaves(grads)
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves)

    def test_is_jittable(self, setup):
        jitted = jax.jit(ppo_loss, static_argnums=(1,))
        loss = jitted(setup["params"], setup["model"], setup["obs"],
                      setup["actions"], setup["phases"], setup["masks"],
                      setup["old_log_probs"], setup["advantages"], setup["returns"])
        assert jnp.isfinite(loss)
