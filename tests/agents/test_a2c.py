"""Tests for JAX A2C loss."""
import jax
import jax.numpy as jnp
import pytest
from src.agents.a2c import a2c_loss
from src.agents.ppo import ppo_loss
from src.agents.actor_critic import ActorCritic
from src.env.constants import obs_dim

N_COLS = 6
OBS_DIM = obs_dim(N_COLS)
MAX_ACTIONS = max(32, 13 * N_COLS)


class TestA2CLoss:
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
        loss = a2c_loss(setup["params"], setup["model"], setup["obs"],
                        setup["actions"], setup["phases"], setup["masks"],
                        setup["old_log_probs"], setup["advantages"], setup["returns"])
        assert loss.shape == ()

    def test_loss_is_finite(self, setup):
        loss = a2c_loss(setup["params"], setup["model"], setup["obs"],
                        setup["actions"], setup["phases"], setup["masks"],
                        setup["old_log_probs"], setup["advantages"], setup["returns"])
        assert jnp.isfinite(loss)

    def test_differentiable(self, setup):
        def loss_fn(p):
            return a2c_loss(p, setup["model"], setup["obs"], setup["actions"],
                            setup["phases"], setup["masks"], setup["old_log_probs"],
                            setup["advantages"], setup["returns"])
        grads = jax.grad(loss_fn)(setup["params"])
        leaves = jax.tree.leaves(grads)
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves)

    def test_is_jittable(self, setup):
        jitted = jax.jit(a2c_loss, static_argnums=(1,))
        loss = jitted(setup["params"], setup["model"], setup["obs"],
                      setup["actions"], setup["phases"], setup["masks"],
                      setup["old_log_probs"], setup["advantages"], setup["returns"])
        assert jnp.isfinite(loss)

    def test_no_clipping(self, setup):
        stale_log_probs = -10.0 * jnp.ones(setup["old_log_probs"].shape)
        a2c = a2c_loss(setup["params"], setup["model"], setup["obs"],
                       setup["actions"], setup["phases"], setup["masks"],
                       stale_log_probs, setup["advantages"], setup["returns"])
        ppo = ppo_loss(setup["params"], setup["model"], setup["obs"],
                       setup["actions"], setup["phases"], setup["masks"],
                       stale_log_probs, setup["advantages"], setup["returns"])
        assert not jnp.allclose(a2c, ppo, atol=1e-4)

    def test_entropy_with_masked_actions(self, setup):
        loss_full_mask = a2c_loss(
            setup["params"], setup["model"], setup["obs"],
            setup["actions"], setup["phases"], setup["masks"],
            setup["old_log_probs"], setup["advantages"], setup["returns"],
            entropy_coef=1.0, value_loss_coef=0.0)
        fewer_masks = jnp.concatenate([
            jnp.ones((8, 16), dtype=jnp.bool_),
            jnp.zeros((8, MAX_ACTIONS - 16), dtype=jnp.bool_),
        ], axis=1)
        small_actions = jax.random.randint(jax.random.PRNGKey(3), (8,), 0, 16)
        loss_fewer_mask = a2c_loss(
            setup["params"], setup["model"], setup["obs"],
            small_actions, setup["phases"], fewer_masks,
            setup["old_log_probs"], setup["advantages"], setup["returns"],
            entropy_coef=1.0, value_loss_coef=0.0)
        assert not jnp.allclose(loss_full_mask, loss_fewer_mask, atol=1e-4)


class TestPerPhaseEntropy:
    @pytest.fixture
    def setup(self):
        model = ActorCritic(hidden_dim=32, n_layers=1, n_columns=N_COLS)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
        n_steps = 8
        obs = jax.random.normal(jax.random.PRNGKey(1), (n_steps, OBS_DIM))
        phases = jnp.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=jnp.int32)
        actions = jnp.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=jnp.int32)
        masks = jnp.ones((n_steps, MAX_ACTIONS), dtype=jnp.bool_)
        old_log_probs = -jnp.ones(n_steps)
        advantages = jnp.ones(n_steps)
        returns = jnp.ones(n_steps)
        return dict(model=model, params=params, obs=obs, actions=actions,
                    phases=phases, masks=masks, old_log_probs=old_log_probs,
                    advantages=advantages, returns=returns)

    def test_per_phase_differs_from_uniform(self, setup):
        uniform = a2c_loss(setup["params"], setup["model"], setup["obs"],
                           setup["actions"], setup["phases"], setup["masks"],
                           setup["old_log_probs"], setup["advantages"], setup["returns"],
                           entropy_coef=0.05)
        per_phase = a2c_loss(setup["params"], setup["model"], setup["obs"],
                             setup["actions"], setup["phases"], setup["masks"],
                             setup["old_log_probs"], setup["advantages"], setup["returns"],
                             entropy_coef=0.05,
                             entropy_coef_roll=0.1, entropy_coef_score=0.01)
        assert not jnp.allclose(uniform, per_phase, atol=1e-6)

    def test_per_phase_is_finite(self, setup):
        loss = a2c_loss(setup["params"], setup["model"], setup["obs"],
                        setup["actions"], setup["phases"], setup["masks"],
                        setup["old_log_probs"], setup["advantages"], setup["returns"],
                        entropy_coef_roll=0.1, entropy_coef_score=0.01)
        assert jnp.isfinite(loss)


class TestUpperRegression:
    @pytest.fixture
    def setup(self):
        model = ActorCritic(hidden_dim=32, n_layers=1, n_columns=N_COLS,
                            head_hidden_dim=32)
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
        upper_targets = jnp.zeros(n_steps)
        return dict(model=model, params=params, obs=obs, actions=actions,
                    phases=phases, masks=masks, old_log_probs=old_log_probs,
                    advantages=advantages, returns=returns,
                    upper_targets=upper_targets)

    def test_upper_regression_changes_loss(self, setup):
        loss_no_upper = a2c_loss(
            setup["params"], setup["model"], setup["obs"],
            setup["actions"], setup["phases"], setup["masks"],
            setup["old_log_probs"], setup["advantages"], setup["returns"])
        loss_with_upper = a2c_loss(
            setup["params"], setup["model"], setup["obs"],
            setup["actions"], setup["phases"], setup["masks"],
            setup["old_log_probs"], setup["advantages"], setup["returns"],
            upper_targets=setup["upper_targets"], upper_regression_weight=1.0)
        assert not jnp.allclose(loss_no_upper, loss_with_upper, atol=1e-6)

    def test_upper_regression_zero_weight_no_effect(self, setup):
        loss_base = a2c_loss(
            setup["params"], setup["model"], setup["obs"],
            setup["actions"], setup["phases"], setup["masks"],
            setup["old_log_probs"], setup["advantages"], setup["returns"])
        loss_zero_w = a2c_loss(
            setup["params"], setup["model"], setup["obs"],
            setup["actions"], setup["phases"], setup["masks"],
            setup["old_log_probs"], setup["advantages"], setup["returns"],
            upper_targets=setup["upper_targets"], upper_regression_weight=0.0)
        assert jnp.allclose(loss_base, loss_zero_w, atol=1e-6)
