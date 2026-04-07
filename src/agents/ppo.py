"""JAX PPO loss. All functions are pure and jit-compatible."""
import jax
import jax.numpy as jnp

from src.agents.common import compute_gae  # noqa: F401 — re-export for backward compat


def ppo_loss(params, model, obs, actions, phases, masks,
             old_log_probs, advantages, returns,
             clip_epsilon=0.2, entropy_coef=0.01, value_loss_coef=0.5):
    logits, values = jax.vmap(model.apply, in_axes=(None, 0, 0))(params, obs, phases)
    logits = jnp.where(masks, logits, -jnp.inf)
    log_probs = jax.nn.log_softmax(logits)
    action_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze(1)
    probs = jax.nn.softmax(logits)
    # Guard against 0 * -inf = NaN for masked-out (zero-probability) actions.
    # Replace -inf log_probs with 0 before multiplying; masked probs are already 0.
    safe_log_probs = jnp.where(masks, log_probs, 0.0)
    entropy = -jnp.sum(probs * safe_log_probs, axis=-1)
    ratio = jnp.exp(action_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -jnp.minimum(surr1, surr2).mean()
    value_loss = 0.5 * jnp.mean((values - returns) ** 2)
    entropy_loss = -entropy_coef * entropy.mean()
    return policy_loss + value_loss_coef * value_loss + entropy_loss
