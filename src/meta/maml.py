"""FOMAML outer loop — meta-gradient aggregation and optimizer step."""
import jax
import jax.numpy as jnp
import optax

from src.meta.inner_loop import collect_episodes, prepare_ppo_data, inner_update_and_query_grad
from src.tasks.reward_tasks import TASK_NAMES


class FOMAML:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.n_inner_steps = config["meta"]["n_inner_steps"]
        self.inner_lr = config["meta"]["inner_lr"]
        self.n_parallel_envs = config["meta"]["n_parallel_envs"]
        self.n_columns = config["env"]["n_columns"]
        self.max_steps = config["env"]["max_steps_per_episode"]
        self.n_tasks = config["meta"]["n_tasks_per_batch"]
        self.gae_lambda = config["ppo"]["gae_lambda"]
        self.optimizer = optax.adam(config["meta"]["outer_lr"])
        self._inner_config = {"inner_lr": self.inner_lr, "n_inner_steps": self.n_inner_steps}

    def init_optimizer(self, params):
        return self.optimizer.init(params)

    def meta_update(self, meta_params, opt_state, rng):
        """One meta-step. Returns (meta_params, opt_state, rng, meta_loss, per_task_losses, skipped)."""
        meta_grads = None
        task_losses = []
        task_ids = list(range(min(self.n_tasks, len(TASK_NAMES))))

        for task_id in task_ids:
            rng, support_rng, query_rng = jax.random.split(rng, 3)
            support_traj = collect_episodes(
                meta_params, self.model, support_rng, jnp.int32(task_id),
                self.n_parallel_envs, self.n_columns, self.max_steps)
            support_data = prepare_ppo_data(support_traj, gamma=0.99, lam=self.gae_lambda)
            query_traj = collect_episodes(
                meta_params, self.model, query_rng, jnp.int32(task_id),
                self.n_parallel_envs, self.n_columns, self.max_steps)
            query_data = prepare_ppo_data(query_traj, gamma=0.99, lam=self.gae_lambda)
            task_grads, query_loss, inner_losses = inner_update_and_query_grad(
                meta_params, self.model, support_data, query_data, self._inner_config)
            if meta_grads is None:
                meta_grads = task_grads
            else:
                meta_grads = jax.tree.map(lambda a, b: a + b, meta_grads, task_grads)
            task_losses.append(float(query_loss))

        n_tasks = len(task_ids)
        meta_grads = jax.tree.map(lambda g: g / n_tasks, meta_grads)

        has_nan = jax.tree_util.tree_reduce(
            lambda acc, g: acc | jnp.any(jnp.isnan(g)) | jnp.any(jnp.isinf(g)),
            meta_grads, initializer=False)
        if has_nan:
            return meta_params, opt_state, rng, float("nan"), task_losses, True

        grad_norm = jnp.sqrt(jax.tree_util.tree_reduce(
            lambda acc, g: acc + jnp.sum(g ** 2), meta_grads, initializer=0.0))
        scale = jnp.minimum(1.0, 1.0 / (grad_norm + 1e-8))
        meta_grads = jax.tree.map(lambda g: g * scale, meta_grads)

        updates, opt_state = self.optimizer.update(meta_grads, opt_state, meta_params)
        meta_params = optax.apply_updates(meta_params, updates)
        meta_loss = sum(task_losses) / n_tasks
        return meta_params, opt_state, rng, meta_loss, task_losses, False
