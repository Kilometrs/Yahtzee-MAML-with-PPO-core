"""JAX PPO loss and GAE computation. All functions are pure and jit-compatible."""
import jax
import jax.numpy as jnp


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    next_values = jnp.concatenate([values[1:], jnp.zeros(1)])

    def scan_fn(last_adv, t):
        reward, value, next_value, done = t
        delta = reward + gamma * next_value * (1.0 - done) - value
        adv = delta + gamma * lam * (1.0 - done) * last_adv
        return adv, adv

    _, advantages = jax.lax.scan(
        scan_fn, 0.0,
        (rewards[::-1], values[::-1], next_values[::-1], dones[::-1]),
    )
    advantages = advantages[::-1]
    returns = advantages + values
    return advantages, returns


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
