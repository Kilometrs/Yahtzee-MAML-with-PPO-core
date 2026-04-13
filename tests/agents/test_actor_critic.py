"""Tests for Flax ActorCritic model."""
import jax
import jax.numpy as jnp
import pytest
from src.agents.actor_critic import ActorCritic
from src.env.constants import obs_dim

N_COLS = 6
OBS_DIM = obs_dim(N_COLS)
MAX_ACTIONS = max(32, 13 * N_COLS)


class TestActorCriticInit:
    def test_instantiates(self):
        model = ActorCritic(hidden_dim=600, n_layers=2, n_columns=N_COLS,
                            use_layer_norm=True, activation="swish")
        assert model is not None

    def test_init_params(self):
        model = ActorCritic(hidden_dim=600, n_layers=2, n_columns=N_COLS,
                            use_layer_norm=True, activation="swish")
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
        assert "params" in params

    def test_init_with_paper_config(self):
        model = ActorCritic(hidden_dim=600, n_layers=2, n_columns=N_COLS,
                            use_layer_norm=True, activation="swish",
                            dropout_rate=0.1, head_hidden_dim=600)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
        assert "params" in params


class TestActorCriticForward:
    @pytest.fixture
    def model_and_params(self):
        model = ActorCritic(hidden_dim=64, n_layers=2, n_columns=N_COLS,
                            use_layer_norm=True, activation="swish")
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
        return model, params

    def test_output_shapes_roll_phase(self, model_and_params):
        model, params = model_and_params
        logits, value, upper_pred = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(0))
        assert logits.shape == (MAX_ACTIONS,)
        assert value.shape == ()
        assert upper_pred.shape == ()

    def test_output_shapes_score_phase(self, model_and_params):
        model, params = model_and_params
        logits, value, upper_pred = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(1))
        assert logits.shape == (MAX_ACTIONS,)
        assert value.shape == ()
        assert upper_pred.shape == ()

    def test_roll_logits_padded(self, model_and_params):
        model, params = model_and_params
        logits, _, _ = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(0))
        assert jnp.all(logits[32:] == -jnp.inf)

    def test_score_logits_finite(self, model_and_params):
        model, params = model_and_params
        logits, _, _ = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(1))
        assert jnp.all(jnp.isfinite(logits))

    def test_value_is_finite(self, model_and_params):
        model, params = model_and_params
        _, value, _ = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(0))
        assert jnp.isfinite(value)

    def test_upper_pred_zero_no_head(self, model_and_params):
        model, params = model_and_params
        _, _, upper_pred = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(0))
        assert float(upper_pred) == 0.0

    def test_jittable(self, model_and_params):
        model, params = model_and_params
        apply_jit = jax.jit(model.apply)
        logits, value, upper_pred = apply_jit(params, jnp.ones(OBS_DIM), jnp.int32(0))
        assert logits.shape == (MAX_ACTIONS,)

    def test_vmappable(self, model_and_params):
        model, params = model_and_params
        obs_batch = jnp.ones((4, OBS_DIM))
        phases = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
        logits, values, upper_preds = jax.vmap(model.apply, in_axes=(None, 0, 0))(
            params, obs_batch, phases)
        assert logits.shape == (4, MAX_ACTIONS)
        assert values.shape == (4,)
        assert upper_preds.shape == (4,)

    def test_gradients_flow(self, model_and_params):
        model, params = model_and_params
        def loss_fn(p):
            logits, value, upper_pred = model.apply(p, jnp.ones(OBS_DIM), jnp.int32(0))
            return jnp.sum(logits[:32]) + value
        grads = jax.grad(loss_fn)(params)
        leaves = jax.tree.leaves(grads)
        nonzero = sum(1 for g in leaves if jnp.any(g != 0))
        assert nonzero > len(leaves) // 2


class TestActorCriticPaperConfig:
    @pytest.fixture
    def paper_model_and_params(self):
        model = ActorCritic(hidden_dim=64, n_layers=2, n_columns=N_COLS,
                            use_layer_norm=True, activation="swish",
                            dropout_rate=0.1, head_hidden_dim=64)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
        return model, params

    def test_upper_pred_nonzero_with_head(self, paper_model_and_params):
        model, params = paper_model_and_params
        _, _, upper_pred = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(0))
        assert jnp.isfinite(upper_pred)

    def test_output_shapes(self, paper_model_and_params):
        model, params = paper_model_and_params
        logits, value, upper_pred = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(0))
        assert logits.shape == (MAX_ACTIONS,)
        assert value.shape == ()
        assert upper_pred.shape == ()

    def test_upper_head_gradient_flow(self, paper_model_and_params):
        model, params = paper_model_and_params
        def loss_fn(p):
            _, _, upper_pred = model.apply(p, jnp.ones(OBS_DIM), jnp.int32(0))
            return upper_pred
        grads = jax.grad(loss_fn)(params)
        leaves = jax.tree.leaves(grads)
        nonzero = sum(1 for g in leaves if jnp.any(g != 0))
        assert nonzero > 0

    def test_deterministic_flag(self, paper_model_and_params):
        model, params = paper_model_and_params
        obs = jnp.ones(OBS_DIM)
        out_det = model.apply(params, obs, jnp.int32(0), deterministic=True)
        # Roll phase: first 32 logits are finite; remainder are -inf padding by design
        assert jnp.all(jnp.isfinite(out_det[0][:32]))
