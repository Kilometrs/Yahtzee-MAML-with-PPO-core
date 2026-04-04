"""FOMAML inner loop and episode collection in JAX.

Three JIT-compiled units:
1. collect_all_episodes — batched over tasks, vmap'd over envs, scan over steps
2. prepare_all_ppo_data — batched GAE + flatten for all tasks
3. all_inner_updates_and_query_grads — batched inner adaptation + query gradients
"""
import functools
import jax
import jax.numpy as jnp

from src.env.yahtzee_env import env_reset, env_step, make_obs, get_action_mask
from src.agents.ppo import compute_gae, ppo_loss


def _collect_episodes_single(params, model, rng_key, task_id,
                             n_parallel_envs, n_columns, max_steps):
    """Collect trajectories for ONE task. Not jit'd — called inside batched wrapper."""
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
        action_log_probs = jnp.take_along_axis(
            log_probs, actions[:, None], axis=1
        ).squeeze(1)

        new_states, _, rewards, dones, _ = jax.vmap(
            env_step, in_axes=(0, 0, None, None)
        )(states, actions, task_id, n_columns)

        reset_keys = jax.random.split(reset_rng, n_parallel_envs)
        reset_states, _ = jax.vmap(env_reset, in_axes=(0, None))(
            reset_keys, n_columns
        )
        next_states = jax.tree.map(
            lambda new, rst: jnp.where(
                jnp.broadcast_to(
                    dones.reshape(-1, *((1,) * (new.ndim - 1))), new.shape
                ),
                rst, new,
            ),
            new_states, reset_states,
        )

        step_data = (obs, actions, action_log_probs, rewards, dones,
                     values, phases, masks)
        return (next_states, rng), step_data

    _, trajectories = jax.lax.scan(
        scan_step, (states, rng_key), None, length=max_steps
    )
    return trajectories


@functools.partial(jax.jit, static_argnums=(1, 4, 5, 6))
def collect_episodes(params, model, rng_key, task_id,
                     n_parallel_envs, n_columns, max_steps):
    """Collect episodes for a single task (backward compat)."""
    return _collect_episodes_single(
        params, model, rng_key, task_id, n_parallel_envs, n_columns, max_steps
    )


@functools.partial(jax.jit, static_argnums=(1, 5, 6, 7))
def collect_all_episodes(params, model, support_rngs, query_rngs, task_ids,
                         n_parallel_envs, n_columns, max_steps):
    """Collect support + query episodes for all tasks in one JIT call.

    Args:
        support_rngs: (n_tasks,) PRNGKeys
        query_rngs: (n_tasks,) PRNGKeys
        task_ids: (n_tasks,) int32
    Returns:
        support_trajs: tuple of (max_steps, n_tasks, n_parallel_envs, ...) arrays
        query_trajs: same shape
    """

    def collect_one(rng, tid):
        return _collect_episodes_single(
            params, model, rng, tid, n_parallel_envs, n_columns, max_steps
        )

    support_trajs = jax.vmap(collect_one)(support_rngs, task_ids)
    query_trajs = jax.vmap(collect_one)(query_rngs, task_ids)
    return support_trajs, query_trajs


def _prepare_one(traj):
    """Prepare PPO data for one task's trajectory."""
    obs, actions, log_probs, rewards, dones, values, phases, masks = traj
    max_steps, n_envs = rewards.shape

    batched_gae = jax.vmap(compute_gae, in_axes=(1, 1, 1, None, None), out_axes=1)
    advantages, returns = batched_gae(rewards, values, dones, 0.99, 0.95)

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


@jax.jit
def prepare_ppo_data(trajectories, gamma=0.99, lam=0.95):
    """Prepare PPO data for a single task trajectory (backward compat)."""
    return _prepare_one(trajectories)


@jax.jit
def prepare_all_ppo_data(support_trajs, query_trajs):
    """Prepare PPO data for all tasks at once.

    Args:
        support_trajs: tuple of (max_steps, n_tasks, n_envs, ...) from collect_all_episodes
        query_trajs: same
    Returns:
        support_data: tuple of (n_tasks, T, ...) arrays
        query_data: same
    """
    support_data = jax.vmap(_prepare_one)(support_trajs)
    query_data = jax.vmap(_prepare_one)(query_trajs)
    return support_data, query_data


def _inner_update_single(meta_params, model, support_data, query_data,
                         inner_lr, n_inner_steps):
    """Inner adaptation + query grad for one task."""
    obs_s, actions_s, phases_s, masks_s, lp_s, adv_s, ret_s = support_data

    fast_params = jax.tree.map(jnp.copy, meta_params)

    def inner_step(fast_params, _):
        loss, grads = jax.value_and_grad(ppo_loss)(
            fast_params, model, obs_s, actions_s, phases_s,
            masks_s, lp_s, adv_s, ret_s,
        )
        grads = jax.lax.stop_gradient(grads)
        grad_norm = jnp.sqrt(jax.tree_util.tree_reduce(
            lambda acc, g: acc + jnp.sum(g ** 2), grads, initializer=0.0
        ))
        scale = jnp.minimum(1.0, 1.0 / (grad_norm + 1e-8))
        grads = jax.tree.map(lambda g: g * scale, grads)
        fast_params = jax.tree.map(
            lambda p, g: p - inner_lr * g, fast_params, grads
        )
        return fast_params, loss

    fast_params, inner_losses = jax.lax.scan(
        inner_step, fast_params, None, length=n_inner_steps
    )

    obs_q, actions_q, phases_q, masks_q, lp_q, adv_q, ret_q = query_data
    query_loss, query_grads = jax.value_and_grad(ppo_loss)(
        fast_params, model, obs_q, actions_q, phases_q,
        masks_q, lp_q, adv_q, ret_q,
    )
    return query_grads, query_loss, inner_losses


@functools.partial(jax.jit, static_argnums=(1, 4, 5))
def _inner_update_and_query_grad_jit(meta_params, model, support_data, query_data,
                                     inner_lr, n_inner_steps):
    """Single-task inner update (backward compat)."""
    return _inner_update_single(
        meta_params, model, support_data, query_data, inner_lr, n_inner_steps
    )


def inner_update_and_query_grad(meta_params, model, support_data, query_data, config):
    """Run inner adaptation, return query gradient. FOMAML (first-order)."""
    inner_lr = float(config["inner_lr"])
    n_inner_steps = int(config["n_inner_steps"])
    return _inner_update_and_query_grad_jit(
        meta_params, model, support_data, query_data, inner_lr, n_inner_steps
    )


@functools.partial(jax.jit, static_argnums=(1, 4, 5))
def all_inner_updates(meta_params, model, all_support_data, all_query_data,
                      inner_lr, n_inner_steps):
    """Run inner updates for ALL tasks at once via vmap.

    Args:
        all_support_data: tuple of (n_tasks, T, ...) arrays
        all_query_data: same
    Returns:
        all_grads: pytree with leading (n_tasks,) dimension
        all_query_losses: (n_tasks,)
        all_inner_losses: (n_tasks, n_inner_steps)
    """
    def per_task(s_data, q_data):
        return _inner_update_single(
            meta_params, model, s_data, q_data, inner_lr, n_inner_steps
        )

    return jax.vmap(per_task)(all_support_data, all_query_data)
