"""Flax linen ActorCritic for Yahtzee FOMAML.

Shared MLP trunk with configurable activation (Swish default) and optional
LayerNorm, three heads: roll logits (32), score logits (13*n_cols), and
value (scalar with ELU activation). Phase-dependent head selection via
jax.lax.cond. Output logits are always padded to max(32, 13*n_columns)
for uniform shapes.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn


class ActorCritic(nn.Module):
    hidden_dim: int = 600
    n_layers: int = 2
    n_columns: int = 6
    use_layer_norm: bool = True
    activation: str = "swish"
    dropout_rate: float = 0.0
    head_hidden_dim: int = 0
    norm_position: str = "pre"

    @nn.compact
    def __call__(self, obs: jnp.ndarray, phase: jnp.int32,
                 deterministic: bool = True):
        max_actions = max(32, 13 * self.n_columns)
        act_fn = nn.swish if self.activation == "swish" else nn.relu

        x = obs
        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden_dim)(x)
            if self.norm_position == "pre" and self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = act_fn(x)
            if self.norm_position == "post" and self.use_layer_norm:
                x = nn.LayerNorm()(x)
            if self.dropout_rate > 0.0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        if self.head_hidden_dim > 0:
            roll_h = act_fn(nn.Dense(self.head_hidden_dim, name="roll_hidden")(x))
            if self.use_layer_norm:
                roll_h = nn.LayerNorm(name="roll_ln")(roll_h)
            roll_logits = nn.Dense(32, name="roll_out")(roll_h)

            score_h = act_fn(nn.Dense(self.head_hidden_dim, name="score_hidden")(x))
            if self.use_layer_norm:
                score_h = nn.LayerNorm(name="score_ln")(score_h)
            score_logits = nn.Dense(13 * self.n_columns, name="score_out")(score_h)

            upper_h = act_fn(nn.Dense(self.head_hidden_dim, name="upper_hidden")(x))
            if self.use_layer_norm:
                upper_h = nn.LayerNorm(name="upper_ln")(upper_h)
            upper_pred = nn.Dense(1, name="upper_out")(upper_h).squeeze(-1)
        else:
            roll_logits = nn.Dense(32)(x)
            score_logits = nn.Dense(13 * self.n_columns)(x)
            upper_pred = jnp.float32(0.0)

        value = nn.elu(nn.Dense(1)(x)).squeeze(-1)

        roll_padded = jnp.concatenate([roll_logits, jnp.full(max_actions - 32, -jnp.inf)])
        score_padded = score_logits
        if 13 * self.n_columns < max_actions:
            score_padded = jnp.concatenate([score_logits, jnp.full(max_actions - 13 * self.n_columns, -jnp.inf)])

        logits = jax.lax.cond(phase == 0, lambda: roll_padded, lambda: score_padded)
        return logits, value, upper_pred
