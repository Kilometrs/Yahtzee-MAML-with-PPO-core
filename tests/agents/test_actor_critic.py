"""Tests for Flax ActorCritic model."""
import jax
import jax.numpy as jnp
import pytest
from src.agents.actor_critic import ActorCritic

N_COLS = 6
OBS_DIM = 7 + 26 * N_COLS
MAX_ACTIONS = max(32, 13 * N_COLS)


class TestActorCriticInit:
    def test_instantiates(self):
        model = ActorCritic(hidden_dim=256, n_layers=3, n_columns=N_COLS)
        assert model is not None

    def test_init_params(self):
        model = ActorCritic(hidden_dim=256, n_layers=3, n_columns=N_COLS)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
        assert "params" in params


class TestActorCriticForward:
    @pytest.fixture
    def model_and_params(self):
        model = ActorCritic(hidden_dim=64, n_layers=2, n_columns=N_COLS)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
        return model, params

    def test_output_shapes_roll_phase(self, model_and_params):
        model, params = model_and_params
        logits, value = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(0))
        assert logits.shape == (MAX_ACTIONS,)
        assert value.shape == ()

    def test_output_shapes_score_phase(self, model_and_params):
        model, params = model_and_params
        logits, value = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(1))
        assert logits.shape == (MAX_ACTIONS,)
        assert value.shape == ()

    def test_roll_logits_padded(self, model_and_params):
        model, params = model_and_params
        logits, _ = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(0))
        assert jnp.all(logits[32:] == -jnp.inf)

    def test_score_logits_finite(self, model_and_params):
        model, params = model_and_params
        logits, _ = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(1))
        assert jnp.all(jnp.isfinite(logits))

    def test_jittable(self, model_and_params):
        model, params = model_and_params
        apply_jit = jax.jit(model.apply)
        logits, value = apply_jit(params, jnp.ones(OBS_DIM), jnp.int32(0))
        assert logits.shape == (MAX_ACTIONS,)

    def test_vmappable(self, model_and_params):
        model, params = model_and_params
        obs_batch = jnp.ones((4, OBS_DIM))
        phases = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
        logits, values = jax.vmap(model.apply, in_axes=(None, 0, 0))(params, obs_batch, phases)
        assert logits.shape == (4, MAX_ACTIONS)
        assert values.shape == (4,)

    def test_gradients_flow(self, model_and_params):
        model, params = model_and_params
        def loss_fn(p):
            logits, value = model.apply(p, jnp.ones(OBS_DIM), jnp.int32(0))
            return jnp.sum(logits[:32]) + value
        grads = jax.grad(loss_fn)(params)
        leaves = jax.tree.leaves(grads)
        assert all(jnp.any(g != 0) for g in leaves)
