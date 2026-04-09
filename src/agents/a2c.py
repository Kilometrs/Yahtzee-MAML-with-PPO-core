"""JAX A2C (Advantage Actor-Critic) loss. All functions are pure and jit-compatible.

Implements vanilla policy gradient with TD(0) advantages following
Pape 2025 (arXiv 2601.00007) Section 4.6.3. No importance ratio or
clipping — single-pass on-policy loss.
"""
import jax
import jax.numpy as jnp


def a2c_loss(params, model, obs, actions, phases, masks,
             old_log_probs, advantages, returns,
             entropy_coef=0.01, value_loss_coef=0.5,
             entropy_coef_roll=None, entropy_coef_score=None,
             upper_targets=None, upper_regression_weight=0.0,
             deterministic=True, rng=None):
    """A2C loss with optional upper score regression (Pape 2025 Section 4.5.1).

    old_log_probs is accepted but unused — keeps the inner loop
    algorithm-agnostic.

    Args:
        entropy_coef_roll: Entropy coefficient for roll steps. If None, uses entropy_coef.
        entropy_coef_score: Entropy coefficient for score steps. If None, uses entropy_coef.
        upper_targets: Normalized final upper scores per timestep. Shape (batch,).
            Target is (upper_final / 63) - 1, range [-1, 5/3].
            None disables the regression loss.
        upper_regression_weight: Coefficient for upper score regression loss.
        deterministic: If False, enables dropout (requires rng).
        rng: PRNGKey for dropout. Required when deterministic=False and model has dropout.
    """
    if deterministic or rng is None:
        logits, values, upper_preds = jax.vmap(model.apply, in_axes=(None, 0, 0))(
            params, obs, phases)
    else:
        rngs = jax.random.split(rng, obs.shape[0])
        def apply_with_dropout(obs_i, phase_i, rng_i):
            return model.apply(params, obs_i, phase_i, deterministic=False,
                               rngs={"dropout": rng_i})
        logits, values, upper_preds = jax.vmap(apply_with_dropout)(obs, phases, rngs)

    logits = jnp.where(masks, logits, -jnp.inf)
    log_probs = jax.nn.log_softmax(logits)
    action_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze(1)
    probs = jax.nn.softmax(logits)
    safe_log_probs = jnp.where(masks, log_probs, 0.0)
    entropy = -jnp.sum(probs * safe_log_probs, axis=-1)

    policy_loss = -(action_log_probs * advantages).mean()
    value_loss = jnp.mean((values - returns) ** 2)

    eff_roll = entropy_coef if entropy_coef_roll is None else entropy_coef_roll
    eff_score = entropy_coef if entropy_coef_score is None else entropy_coef_score

    is_roll = (phases == 0).astype(jnp.float32)
    is_score = (phases == 1).astype(jnp.float32)
    n_roll = jnp.maximum(is_roll.sum(), 1.0)
    n_score = jnp.maximum(is_score.sum(), 1.0)
    roll_entropy = (entropy * is_roll).sum() / n_roll
    score_entropy = (entropy * is_score).sum() / n_score
    entropy_loss = -(eff_roll * roll_entropy + eff_score * score_entropy)

    upper_loss = jnp.where(
        upper_regression_weight > 0.0,
        upper_regression_weight * jnp.mean((upper_preds - upper_targets) ** 2),
        0.0,
    ) if upper_targets is not None else 0.0

    return policy_loss + value_loss_coef * value_loss + entropy_loss + upper_loss
