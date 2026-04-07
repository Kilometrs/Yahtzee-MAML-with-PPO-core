"""FOMAML outer loop — meta-gradient aggregation and optimizer step."""
import jax
import jax.numpy as jnp
import optax

from src.agents.a2c import a2c_loss
from src.agents.ppo import ppo_loss
from src.meta.inner_loop import (
    collect_all_episodes, prepare_all_ppo_data, all_inner_updates,
    # backward compat for evaluator / tests
    collect_episodes, prepare_ppo_data, inner_update_and_query_grad,
)
from src.tasks.reward_tasks import TASK_NAMES


class FOMAML:
    def __init__(self, model, config):
        self.model = model
        self.config = config

        if "a2c" in config:
            self.loss_fn = a2c_loss
            algo_config = config["a2c"]
        elif "ppo" in config:
            self.loss_fn = ppo_loss
            algo_config = config["ppo"]
        else:
            raise ValueError("Config must contain either 'a2c' or 'ppo' section")

        self.inner_lr = float(config["meta"]["inner_lr"])
        self.n_inner_steps = int(config["meta"]["n_inner_steps"])
        self.n_parallel_envs = config["meta"]["n_parallel_envs"]
        self.n_columns = config["env"]["n_columns"]
        self.max_steps = config["env"]["max_steps_per_episode"]
        self.n_tasks = min(config["meta"]["n_tasks_per_batch"], len(TASK_NAMES))
        self.threshold = config["tasks"]["threshold_beater_score"]
        self.gae_lambda = algo_config.get("gae_lambda", 0.95)
        self.optimizer = optax.adam(config["meta"]["outer_lr"])

        self._task_ids = jnp.arange(self.n_tasks, dtype=jnp.int32)
        self._inner_config = {
            "inner_lr": self.inner_lr,
            "n_inner_steps": self.n_inner_steps,
        }

    def init_optimizer(self, params):
        return self.optimizer.init(params)

    def meta_update(self, meta_params, opt_state, rng):
        """One meta-step — fully batched over tasks.

        Returns (meta_params, opt_state, rng, meta_loss, per_task_losses, skipped).
        """
        n_tasks = self.n_tasks

        # Split RNG: n_tasks for support + n_tasks for query
        rng, split_rng = jax.random.split(rng)
        all_rngs = jax.random.split(split_rng, 2 * n_tasks)
        support_rngs = all_rngs[:n_tasks]
        query_rngs = all_rngs[n_tasks:]

        # 1. Collect all episodes (batched over tasks)
        support_trajs, query_trajs = collect_all_episodes(
            meta_params, self.model, support_rngs, query_rngs, self._task_ids,
            self.n_parallel_envs, self.n_columns, self.max_steps,
            self.threshold,
        )

        # 2. Prepare PPO data for all tasks (batched)
        all_support_data, all_query_data = prepare_all_ppo_data(
            support_trajs, query_trajs,
        )

        # 3. Inner updates for all tasks (batched via vmap)
        all_grads, all_query_losses, all_inner_losses = all_inner_updates(
            meta_params, self.model, self.loss_fn, all_support_data, all_query_data,
            self.inner_lr, self.n_inner_steps,
        )

        # 4. Average gradients across tasks
        meta_grads = jax.tree.map(lambda g: jnp.mean(g, axis=0), all_grads)

        # Task losses for logging
        task_losses = [float(all_query_losses[i]) for i in range(n_tasks)]

        # NaN guard
        has_nan = jax.tree_util.tree_reduce(
            lambda acc, g: acc | jnp.any(jnp.isnan(g)) | jnp.any(jnp.isinf(g)),
            meta_grads, initializer=False,
        )
        if has_nan:
            return meta_params, opt_state, rng, float("nan"), task_losses, True

        # Clip meta-gradient (max_norm=1.0)
        grad_norm = jnp.sqrt(jax.tree_util.tree_reduce(
            lambda acc, g: acc + jnp.sum(g ** 2), meta_grads, initializer=0.0,
        ))
        scale = jnp.minimum(1.0, 1.0 / (grad_norm + 1e-8))
        meta_grads = jax.tree.map(lambda g: g * scale, meta_grads)

        # Optimizer step
        updates, opt_state = self.optimizer.update(
            meta_grads, opt_state, meta_params,
        )
        meta_params = optax.apply_updates(meta_params, updates)

        meta_loss = sum(task_losses) / n_tasks
        return meta_params, opt_state, rng, meta_loss, task_losses, False
