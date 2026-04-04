"""Flax linen ActorCritic for Yahtzee FOMAML.

Shared MLP trunk with three heads: roll logits (32), score logits (13*n_cols),
and value (scalar). Phase-dependent head selection via jax.lax.cond.
Output logits are always padded to max(32, 13*n_columns) for uniform shapes.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn


class ActorCritic(nn.Module):
    hidden_dim: int = 256
    n_layers: int = 3
    n_columns: int = 6

    @nn.compact
    def __call__(self, obs: jnp.ndarray, phase: jnp.int32):
        max_actions = max(32, 13 * self.n_columns)

        x = obs
        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)

        roll_logits = nn.Dense(32)(x)
        score_logits = nn.Dense(13 * self.n_columns)(x)
        value = nn.Dense(1)(x).squeeze(-1)

        roll_padded = jnp.concatenate([roll_logits, jnp.full(max_actions - 32, -jnp.inf)])
        score_padded = score_logits
        if 13 * self.n_columns < max_actions:
            score_padded = jnp.concatenate([score_logits, jnp.full(max_actions - 13 * self.n_columns, -jnp.inf)])

        logits = jax.lax.cond(phase == 0, lambda: roll_padded, lambda: score_padded)
        return logits, value
