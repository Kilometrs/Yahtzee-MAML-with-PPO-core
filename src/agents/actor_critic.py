"""Flax linen ActorCritic for Yahtzee.

Architecture matches Pape 2025 (arXiv 2601.00007) Figure 1:
  Trunk: n_layers Blocks (Linear → Activation → LayerNorm → Dropout)
  Heads: each has a Block(head_hidden_dim) → Linear(output)
    - RollingHead  → 32 logits
    - ScoringHead  → 13*n_cols logits (masked)
    - ValueHead    → scalar (ELU activation)
    - UpperHead    → scalar (no activation, predicts normalized upper score)

Weight init: Kaiming normal for hidden layers, orthogonal (gain=0.01) for
output layers (paper Section 4.4.1).
"""
import jax
import jax.numpy as jnp
from flax import linen as nn

# Paper-aligned initializers
_kaiming_init = nn.initializers.he_normal()
_orthogonal_init = nn.initializers.orthogonal(scale=0.01)


class ActorCritic(nn.Module):
    hidden_dim: int = 600
    n_layers: int = 2
    n_columns: int = 6
    use_layer_norm: bool = True
    activation: str = "swish"
    dropout_rate: float = 0.0
    head_hidden_dim: int = 0
    norm_position: str = "pre"

    def _block(self, x, features, name, act_fn, deterministic, dropout_rate=None):
        """One Block: Linear → Activation → LayerNorm → Dropout (paper order)."""
        dr = self.dropout_rate if dropout_rate is None else dropout_rate
        x = nn.Dense(features, kernel_init=_kaiming_init, name=f"{name}_dense")(x)
        x = act_fn(x)
        if self.use_layer_norm:
            x = nn.LayerNorm(name=f"{name}_ln")(x)
        if dr > 0.0:
            x = nn.Dropout(rate=dr)(x, deterministic=deterministic)
        return x

    @nn.compact
    def __call__(self, obs: jnp.ndarray, phase: jnp.int32,
                 deterministic: bool = True):
        max_actions = max(32, 13 * self.n_columns)
        act_fn = nn.swish if self.activation == "swish" else nn.relu

        # Trunk
        x = obs
        for i in range(self.n_layers):
            x = self._block(x, self.hidden_dim, f"trunk_{i}", act_fn, deterministic)

        # Heads — paper: rolling/scoring have dropout, value/upper don't
        if self.head_hidden_dim > 0:
            roll_h = self._block(x, self.head_hidden_dim, "roll_head", act_fn, deterministic)
            roll_logits = nn.Dense(32, kernel_init=_orthogonal_init, name="roll_out")(roll_h)

            score_h = self._block(x, self.head_hidden_dim, "score_head", act_fn, deterministic)
            score_logits = nn.Dense(13 * self.n_columns, kernel_init=_orthogonal_init, name="score_out")(score_h)

            value_h = self._block(x, self.head_hidden_dim, "value_head", act_fn, deterministic, dropout_rate=0.0)
            value = nn.elu(nn.Dense(1, kernel_init=_orthogonal_init, name="value_out")(value_h)).squeeze(-1)

            upper_h = self._block(x, self.head_hidden_dim, "upper_head", act_fn, deterministic, dropout_rate=0.0)
            upper_pred = nn.Dense(1, kernel_init=_orthogonal_init, name="upper_out")(upper_h).squeeze(-1)
        else:
            roll_logits = nn.Dense(32)(x)
            score_logits = nn.Dense(13 * self.n_columns)(x)
            value = nn.elu(nn.Dense(1)(x)).squeeze(-1)
            upper_pred = jnp.float32(0.0)

        roll_padded = jnp.concatenate([roll_logits, jnp.full(max_actions - 32, -jnp.inf)])
        score_padded = score_logits
        if 13 * self.n_columns < max_actions:
            score_padded = jnp.concatenate([score_logits, jnp.full(max_actions - 13 * self.n_columns, -jnp.inf)])

        logits = jax.lax.cond(phase == 0, lambda: roll_padded, lambda: score_padded)
        return logits, value, upper_pred
