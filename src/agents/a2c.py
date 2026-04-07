"""JAX A2C (Advantage Actor-Critic) loss. All functions are pure and jit-compatible.

Implements vanilla policy gradient with TD(0) advantages following
Pape 2025 (arXiv 2601.00007) Section 4.6.3. No importance ratio or
clipping — single-pass on-policy loss.
"""
import jax
import jax.numpy as jnp


def a2c_loss(params, model, obs, actions, phases, masks,
             old_log_probs, advantages, returns,
             entropy_coef=0.01, value_loss_coef=0.5):
    """A2C loss with uniform signature to ppo_loss.

    old_log_probs is accepted but unused — keeps the inner loop
    algorithm-agnostic.
    """
    logits, values = jax.vmap(model.apply, in_axes=(None, 0, 0))(params, obs, phases)
    logits = jnp.where(masks, logits, -jnp.inf)
    log_probs = jax.nn.log_softmax(logits)
    action_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze(1)
    probs = jax.nn.softmax(logits)
    safe_log_probs = jnp.where(masks, log_probs, 0.0)
    entropy = -jnp.sum(probs * safe_log_probs, axis=-1)

    policy_loss = -(action_log_probs * advantages).mean()
    value_loss = 0.5 * jnp.mean((values - returns) ** 2)
    entropy_loss = -entropy_coef * entropy.mean()
    return policy_loss + value_loss_coef * value_loss + entropy_loss
