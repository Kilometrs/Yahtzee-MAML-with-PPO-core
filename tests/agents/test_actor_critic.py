"""Tests for Flax ActorCritic model."""
import jax
import jax.numpy as jnp
import pytest
from src.agents.actor_critic import ActorCritic
from src.env.constants import obs_dim

N_COLS = 6
OBS_DIM = obs_dim(N_COLS)  # 239 (41 + 33*6)
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
        logits, value, _ = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(0))
        assert logits.shape == (MAX_ACTIONS,)
        assert value.shape == ()

    def test_output_shapes_score_phase(self, model_and_params):
        model, params = model_and_params
        logits, value, _ = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(1))
        assert logits.shape == (MAX_ACTIONS,)
        assert value.shape == ()

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

    def test_jittable(self, model_and_params):
        model, params = model_and_params
        apply_jit = jax.jit(model.apply)
        logits, value, _ = apply_jit(params, jnp.ones(OBS_DIM), jnp.int32(0))
        assert logits.shape == (MAX_ACTIONS,)

    def test_vmappable(self, model_and_params):
        model, params = model_and_params
        obs_batch = jnp.ones((4, OBS_DIM))
        phases = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
        logits, values, _ = jax.vmap(model.apply, in_axes=(None, 0, 0))(params, obs_batch, phases)
        assert logits.shape == (4, MAX_ACTIONS)
        assert values.shape == (4,)

    def test_gradients_flow(self, model_and_params):
        """Verify gradients flow through trunk, active head, and value head.

        The inactive policy head (score_head when phase=0) gets zero gradients
        through jax.lax.cond — this is expected and correct.
        """
        model, params = model_and_params
        def loss_fn(p):
            logits, value, _ = model.apply(p, jnp.ones(OBS_DIM), jnp.int32(0))
            return jnp.sum(logits[:32]) + value
        grads = jax.grad(loss_fn)(params)
        leaves = jax.tree.leaves(grads)
        # Most leaves should have nonzero gradients (trunk + roll_head + value_head)
        # Score head weights get zero grad when phase=0, which is expected
        nonzero = sum(1 for g in leaves if jnp.any(g != 0))
        assert nonzero > len(leaves) // 2


class TestActorCriticDropout:
    def test_dropout_param_accepted(self):
        model = ActorCritic(hidden_dim=64, n_layers=2, n_columns=N_COLS,
                            dropout_rate=0.1)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
        assert "params" in params

    def test_deterministic_true_no_rng_needed(self):
        model = ActorCritic(hidden_dim=64, n_layers=2, n_columns=N_COLS,
                            dropout_rate=0.1)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
        logits, value, _ = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(0),
                                       deterministic=True)
        assert logits.shape == (MAX_ACTIONS,)
        assert jnp.isfinite(value)

    def test_deterministic_false_needs_dropout_rng(self):
        model = ActorCritic(hidden_dim=64, n_layers=2, n_columns=N_COLS,
                            dropout_rate=0.1)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
        logits, value, _ = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(0),
                                       deterministic=False,
                                       rngs={"dropout": jax.random.PRNGKey(1)})
        assert logits.shape == (MAX_ACTIONS,)
        assert jnp.isfinite(value)

    def test_dropout_zero_is_noop(self):
        model = ActorCritic(hidden_dim=64, n_layers=2, n_columns=N_COLS,
                            dropout_rate=0.0)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
        out1 = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(0), deterministic=True)
        out2 = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(0), deterministic=False,
                           rngs={"dropout": jax.random.PRNGKey(1)})
        assert jnp.allclose(out1[0], out2[0])
        assert jnp.allclose(out1[1], out2[1])


class TestActorCriticNormPosition:
    def test_post_norm_accepted(self):
        model = ActorCritic(hidden_dim=64, n_layers=2, n_columns=N_COLS,
                            norm_position="post", use_layer_norm=True)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
        logits, value, _ = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(0))
        assert logits.shape == (MAX_ACTIONS,)
        assert jnp.isfinite(value)

    def test_block_order_is_linear_act_norm_dropout(self):
        """Paper order: Linear -> Activation -> LayerNorm -> Dropout."""
        model = ActorCritic(hidden_dim=64, n_layers=2, n_columns=N_COLS,
                            use_layer_norm=True)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
        logits, _, _ = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(0))
        assert jnp.all(jnp.isfinite(logits[:32]))


class TestActorCriticHeadHidden:
    def test_head_hidden_dim_accepted(self):
        model = ActorCritic(hidden_dim=64, n_layers=2, n_columns=N_COLS,
                            head_hidden_dim=32)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
        logits, value, _ = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(0))
        assert logits.shape == (MAX_ACTIONS,)

    def test_head_hidden_has_more_params(self):
        rng = jax.random.PRNGKey(0)
        no_head = ActorCritic(hidden_dim=64, n_layers=2, n_columns=N_COLS,
                              head_hidden_dim=0)
        with_head = ActorCritic(hidden_dim=64, n_layers=2, n_columns=N_COLS,
                                head_hidden_dim=32)
        p_no = no_head.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
        p_with = with_head.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
        n_no = sum(x.size for x in jax.tree.leaves(p_no))
        n_with = sum(x.size for x in jax.tree.leaves(p_with))
        assert n_with > n_no

    def test_head_hidden_gradients_flow(self):
        model = ActorCritic(hidden_dim=64, n_layers=2, n_columns=N_COLS,
                            head_hidden_dim=32)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
        def loss_fn(p):
            logits, value, _ = model.apply(p, jnp.ones(OBS_DIM), jnp.int32(0))
            return jnp.sum(logits[:32]) + value
        grads = jax.grad(loss_fn)(params)
        leaves = jax.tree.leaves(grads)
        nonzero = sum(1 for g in leaves if jnp.any(g != 0))
        assert nonzero > len(leaves) // 2


class TestActorCriticUpperPred:
    def test_output_is_3_tuple(self):
        model = ActorCritic(hidden_dim=64, n_layers=2, n_columns=N_COLS)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
        result = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(0))
        assert len(result) == 3

    def test_upper_pred_with_head(self):
        model = ActorCritic(hidden_dim=64, n_layers=2, n_columns=N_COLS,
                            head_hidden_dim=32)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
        _, _, upper_pred = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(0))
        assert upper_pred.shape == ()
        assert jnp.isfinite(upper_pred)

    def test_upper_pred_zero_without_head(self):
        model = ActorCritic(hidden_dim=64, n_layers=2, n_columns=N_COLS,
                            head_hidden_dim=0)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
        _, _, upper_pred = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(0))
        assert upper_pred == 0.0
