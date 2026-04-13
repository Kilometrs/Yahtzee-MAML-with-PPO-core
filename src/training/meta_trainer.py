"""Meta-training orchestration for JAX FOMAML."""
import os
import numpy as np
import jax
import jax.numpy as jnp
import yaml
from tqdm import tqdm

from src.agents.actor_critic import ActorCritic
from src.env.constants import obs_dim
from src.meta.maml import FOMAML
from src.tasks.reward_tasks import TASK_NAMES
from src.logging_utils import create_logger
from src.training.evaluator import Evaluator


class MetaTrainer:
    def __init__(self, config):
        self.config = config
        self.model = ActorCritic(
            hidden_dim=config["agent"]["hidden_dim"],
            n_layers=config["agent"]["n_layers"],
            n_columns=config["env"]["n_columns"],
            use_layer_norm=config["agent"].get("use_layer_norm", False),
            activation=config["agent"].get("activation", "relu"),
            dropout_rate=config["agent"].get("dropout_rate", 0.0),
            head_hidden_dim=config["agent"].get("head_hidden_dim", 0),
        )
        self.rng = jax.random.PRNGKey(config["env"]["seed"])
        self.rng, init_rng = jax.random.split(self.rng)
        od = obs_dim(config["env"]["n_columns"])
        self.meta_params = self.model.init(init_rng, jnp.zeros(od), jnp.int32(0))
        self.fomaml = FOMAML(self.model, config)
        self.opt_state = self.fomaml.init_optimizer(self.meta_params)
        self.logger = create_logger(
            config["clearml"]["project_name"], config["clearml"]["task_name"], config)
        base_dir = config["training"].get("checkpoint_dir", "checkpoints")
        self.checkpoint_dir = os.path.join(base_dir, self.logger.task_id)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"Checkpoints: {self.checkpoint_dir}")
        self.evaluator = Evaluator(config)
        self.eval_every = config["training"].get("eval_every", 0)
        self.n_eval_episodes = config["training"].get("n_eval_episodes", 50)
        self.log_every = config["training"].get("log_every", 50)

    def train(self, start_step=0):
        n_meta_steps = self.config["meta"]["n_meta_steps"]
        checkpoint_every = self.config["training"]["checkpoint_every"]
        loss_csv = os.path.join(self.checkpoint_dir, "loss.csv")
        if start_step == 0:
            with open(loss_csv, "w") as f:
                f.write("step,meta_loss," + ",".join(TASK_NAMES) + "\n")
        pbar = tqdm(range(start_step, n_meta_steps), initial=start_step, total=n_meta_steps)
        for step in pbar:
            result = self.fomaml.meta_update(self.meta_params, self.opt_state, self.rng)
            self.meta_params, self.opt_state, self.rng, meta_loss, task_losses, skipped = result
            if skipped:
                pbar.set_postfix({"loss": "NaN (skipped)"})
                continue
            pbar.set_postfix({"loss": f"{meta_loss:.4f}"})
            if self.log_every and step % self.log_every == 0:
                self.logger.log_scalar("meta", "loss", meta_loss, step)
                for i, tl in enumerate(task_losses):
                    self.logger.log_scalar("task_loss", TASK_NAMES[i], tl, step)
                with open(loss_csv, "a") as f:
                    f.write(f"{step},{meta_loss}," + ",".join(f"{tl}" for tl in task_losses) + "\n")
            if (step + 1) % checkpoint_every == 0:
                self.save_checkpoint(step + 1)
            if self.eval_every and (step + 1) % self.eval_every == 0:
                self._run_eval(step + 1)

    def save_checkpoint(self, meta_step):
        params_np = jax.device_get(self.meta_params)
        opt_state_np = jax.device_get(self.opt_state)
        path = os.path.join(self.checkpoint_dir, f"step_{meta_step}")
        os.makedirs(path, exist_ok=True)
        # Flatten pytree to flat dict for np.savez
        flat_params = {}
        for k, v in jax.tree_util.tree_leaves_with_path(params_np):
            key = "/".join(str(p) for p in k)
            flat_params[key] = np.asarray(v)
        np.savez(os.path.join(path, "params.npz"), **flat_params)
        # Save opt state similarly
        flat_opt = {}
        for k, v in jax.tree_util.tree_leaves_with_path(opt_state_np):
            key = "/".join(str(p) for p in k)
            flat_opt[key] = np.asarray(v)
        np.savez(os.path.join(path, "opt_state.npz"), **flat_opt)
        np.savez(os.path.join(path, "meta.npz"), meta_step=meta_step, task_names=TASK_NAMES)
        self.logger.log_artifact(f"checkpoint_step{meta_step}", path)
        return path

    def _run_eval(self, meta_step):
        df_steps, df_episodes = self.evaluator.evaluate(
            self.meta_params, meta_step, n_episodes=self.n_eval_episodes)
        mean_score = float(df_episodes["final_score"].mean())
        self.logger.log_scalar("eval", "mean_final_score", mean_score, meta_step)
        for task_name in TASK_NAMES:
            task_mean = float(df_episodes[df_episodes["strategy"] == task_name]["final_score"].mean())
            self.logger.log_scalar("eval_per_task", task_name, task_mean, meta_step)
        self.logger.flush_scalars()
        steps_path, episodes_path = self.evaluator.save_trajectories(
            df_steps, df_episodes, "all", meta_step, out_dir=self.checkpoint_dir)
        self.logger.log_artifact(f"eval_steps_step{meta_step}", steps_path)
        self.logger.log_artifact(f"eval_episodes_step{meta_step}", episodes_path)

    def load_checkpoint(self, path):
        meta_data = np.load(os.path.join(path, "meta.npz"), allow_pickle=True)
        meta_step = int(meta_data["meta_step"])
        # For params: load flat dict and unflatten using tree structure
        params_data = dict(np.load(os.path.join(path, "params.npz"), allow_pickle=True))
        flat_params = [jnp.array(params_data["/".join(str(p) for p in k)])
                       for k, _ in jax.tree_util.tree_leaves_with_path(self.meta_params)]
        self.meta_params = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(self.meta_params), flat_params)
        # Same for opt state
        opt_data = dict(np.load(os.path.join(path, "opt_state.npz"), allow_pickle=True))
        flat_opt = [jnp.array(opt_data["/".join(str(p) for p in k)])
                    for k, _ in jax.tree_util.tree_leaves_with_path(self.opt_state)]
        self.opt_state = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(self.opt_state), flat_opt)
        return meta_step
