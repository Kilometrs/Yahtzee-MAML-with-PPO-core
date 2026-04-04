"""FOMAML inner loop and episode collection in JAX.

Two JIT-compiled units:
1. collect_episodes — vmap'd env stepping with scan
2. inner_update_and_query_grad — inner adaptation + query gradient
"""
import functools
import jax
import jax.numpy as jnp

from src.env.yahtzee_env import env_reset, env_step, make_obs, get_action_mask
from src.agents.ppo import compute_gae, ppo_loss


@functools.partial(jax.jit, static_argnums=(1, 4, 5, 6))
def collect_episodes(params, model, rng_key, task_id,
                     n_parallel_envs, n_columns, max_steps):
    """Collect fixed-length trajectories from parallel envs.
    Returns tuple of arrays, each shape (max_steps, n_parallel_envs, ...)."""
    max_actions = max(32, 13 * n_columns)

    rng_key, init_rng = jax.random.split(rng_key)
    init_keys = jax.random.split(init_rng, n_parallel_envs)
    states, _ = jax.vmap(env_reset, in_axes=(0, None))(init_keys, n_columns)

    def scan_step(carry, _):
        states, rng = carry
        rng, act_rng, reset_rng = jax.random.split(rng, 3)

        obs = jax.vmap(make_obs, in_axes=(0, None))(states, n_columns)
        masks = jax.vmap(get_action_mask, in_axes=(0, None))(states, n_columns)
        phases = states.phase

        logits, values = jax.vmap(model.apply, in_axes=(None, 0, 0))(params, obs, phases)
        logits = jnp.where(masks, logits, -jnp.inf)

        actions = jax.random.categorical(act_rng, logits)
        log_probs = jax.nn.log_softmax(logits)
        action_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze(1)

        new_states, _, rewards, dones, _ = jax.vmap(
            env_step, in_axes=(0, 0, None, None)
        )(states, actions, task_id, n_columns)

        # Auto-reset done envs
        reset_keys = jax.random.split(reset_rng, n_parallel_envs)
        reset_states, _ = jax.vmap(env_reset, in_axes=(0, None))(reset_keys, n_columns)
        next_states = jax.tree.map(
            lambda new, rst: jnp.where(
                jnp.broadcast_to(dones.reshape(-1, *((1,) * (new.ndim - 1))), new.shape),
                rst, new),
            new_states, reset_states)

        step_data = (obs, actions, action_log_probs, rewards, dones, values, phases, masks)
        return (next_states, rng), step_data

    _, trajectories = jax.lax.scan(scan_step, (states, rng_key), None, length=max_steps)
    return trajectories


def prepare_ppo_data(trajectories, gamma=0.99, lam=0.95):
    """Flatten trajectories and compute GAE."""
    obs, actions, log_probs, rewards, dones, values, phases, masks = trajectories
    max_steps, n_envs = rewards.shape

    all_advantages = []
    all_returns = []
    for i in range(n_envs):
        adv, ret = compute_gae(rewards[:, i], values[:, i], dones[:, i], gamma=gamma, lam=lam)
        all_advantages.append(adv)
        all_returns.append(ret)

    advantages = jnp.stack(all_advantages, axis=1)
    returns = jnp.stack(all_returns, axis=1)

    T = max_steps * n_envs
    obs_flat = obs.reshape(T, -1)
    actions_flat = actions.reshape(T)
    phases_flat = phases.reshape(T)
    masks_flat = masks.reshape(T, -1)
    log_probs_flat = log_probs.reshape(T)
    advantages_flat = advantages.reshape(T)
    returns_flat = returns.reshape(T)

    adv_mean = jnp.mean(advantages_flat)
    adv_std = jnp.std(advantages_flat) + 1e-8
    advantages_flat = (advantages_flat - adv_mean) / adv_std

    return (obs_flat, actions_flat, phases_flat, masks_flat,
            log_probs_flat, advantages_flat, returns_flat)


@functools.partial(jax.jit, static_argnums=(1, 4, 5))
def _inner_update_and_query_grad_jit(meta_params, model, support_data, query_data,
                                     inner_lr, n_inner_steps):
    """JIT-compiled inner adaptation + query gradient (FOMAML first-order).

    inner_lr and n_inner_steps are static so they can be used in lax.scan length
    and captured as Python scalars in the closure.
    """
    obs_s, actions_s, phases_s, masks_s, lp_s, adv_s, ret_s = support_data

    fast_params = jax.tree.map(jnp.copy, meta_params)

    def inner_step(fast_params, _):
        loss, grads = jax.value_and_grad(ppo_loss)(
            fast_params, model, obs_s, actions_s, phases_s, masks_s, lp_s, adv_s, ret_s)
        grads = jax.lax.stop_gradient(grads)
        grad_norm = jnp.sqrt(jax.tree_util.tree_reduce(
            lambda acc, g: acc + jnp.sum(g ** 2), grads, initializer=0.0))
        scale = jnp.minimum(1.0, 1.0 / (grad_norm + 1e-8))
        grads = jax.tree.map(lambda g: g * scale, grads)
        fast_params = jax.tree.map(lambda p, g: p - inner_lr * g, fast_params, grads)
        return fast_params, loss

    fast_params, inner_losses = jax.lax.scan(inner_step, fast_params, None, length=n_inner_steps)

    obs_q, actions_q, phases_q, masks_q, lp_q, adv_q, ret_q = query_data
    query_loss, query_grads = jax.value_and_grad(ppo_loss)(
        fast_params, model, obs_q, actions_q, phases_q, masks_q, lp_q, adv_q, ret_q)

    return query_grads, query_loss, inner_losses


def inner_update_and_query_grad(meta_params, model, support_data, query_data, config):
    """Run inner adaptation on support data, return query gradient. FOMAML (first-order).

    Extracts concrete scalar values from config before JIT so that inner_lr and
    n_inner_steps are available as static Python values inside the compiled function.
    """
    inner_lr = float(config["inner_lr"])
    n_inner_steps = int(config["n_inner_steps"])
    return _inner_update_and_query_grad_jit(
        meta_params, model, support_data, query_data, inner_lr, n_inner_steps
    )
