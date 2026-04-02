"""Meta-training orchestration — outer FOMAML loop with checkpointing."""

import random
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from agents.actor_critic import ActorCritic
from env.yahtzee_env import YahtzeeEnv
from meta.maml import FOMAML
from tasks.reward_tasks import TASK_REGISTRY
from logging_utils.clearml_logger import ClearMLLogger
from training.evaluator import Evaluator


class MetaTrainer:
    """Orchestrates the FOMAML outer training loop.

    Responsibilities:
    - Build meta-policy, env factory, task distribution from config
    - Run outer loop for n_meta_steps
    - Save checkpoints every checkpoint_every steps
    - Log scalars and artifacts via ClearMLLogger
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Full config dict loaded from YAML (see configs/default.yaml).
        """
        self.config = config
        self.device = torch.device(config["training"]["device"]
                                   if torch.cuda.is_available()
                                   else "cpu")

        n_columns = config["env"]["n_columns"]
        obs_dim = 7 + 26 * n_columns
        self.model = ActorCritic(
            obs_dim=obs_dim,
            n_columns=n_columns,
            hidden_dim=config["agent"]["hidden_dim"],
            n_layers=config["agent"]["n_layers"],
        ).to(self.device)

        self.fomaml = FOMAML(
            model=self.model,
            inner_lr=config["meta"]["inner_lr"],
            outer_lr=config["meta"]["outer_lr"],
            n_inner_steps=config["meta"]["n_inner_steps"],
        )

        threshold = config["tasks"]["threshold_beater_score"]
        self.tasks = [
            TASK_REGISTRY["MaxScore"](),
            TASK_REGISTRY["ThresholdBeater"](threshold=threshold),
            TASK_REGISTRY["UpperBonus"](),
            TASK_REGISTRY["YahtzeeHunter"](),
            TASK_REGISTRY["Conservative"](),
        ]

        self.n_columns = n_columns
        self.meta_step = 0

        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        n_tasks = config["meta"]["n_tasks_per_batch"]
        auto_task_name = f"{timestamp}_maml_{n_tasks}t"
        task_name = config["clearml"].get("task_name") or auto_task_name

        self.logger = ClearMLLogger(
            project_name=config["clearml"]["project_name"],
            task_name=task_name,
            config=config,
        )
        self.checkpoint_dir = Path(config["training"]["checkpoint_dir"]) / self.logger.task_id
        self.evaluator = Evaluator(config)

    def _env_fn(self) -> YahtzeeEnv:
        return YahtzeeEnv(n_columns=self.n_columns)

    def train(self) -> None:
        """Run full meta-training loop."""
        cfg = self.config
        n_meta_steps = cfg["meta"]["n_meta_steps"]
        n_tasks_per_batch = cfg["meta"]["n_tasks_per_batch"]
        checkpoint_every = cfg["training"]["checkpoint_every"]
        ppo_cfg = cfg["ppo"]

        pbar = tqdm(range(self.meta_step, n_meta_steps), initial=self.meta_step, total=n_meta_steps)
        for step in pbar:
            task_batch = random.sample(self.tasks, min(n_tasks_per_batch, len(self.tasks)))
            meta_loss, per_task_losses = self.fomaml.meta_update(
                tasks=task_batch,
                env_fn=self._env_fn,
                ppo_cfg=ppo_cfg,
                device=self.device,
            )
            self.meta_step = step + 1

            pbar.set_postfix({"loss": f"{meta_loss:.4f}"})
            self.logger.log_scalar("Loss", "meta_loss", meta_loss, step)
            for task_name, task_loss in per_task_losses.items():
                self.logger.log_scalar("Loss/task", task_name, task_loss, step)

            if self.meta_step % checkpoint_every == 0:
                ckpt_path = self.save_checkpoint(self.meta_step)
                self.logger.log_artifact(
                    name=f"checkpoint_step{self.meta_step}",
                    path=str(ckpt_path),
                )
                df_steps, df_episodes = self.evaluator.evaluate(
                    checkpoint_path=ckpt_path,
                    n_episodes=20,
                )
                for task in self.tasks:
                    task_steps = df_steps[df_steps["strategy"] == task.name]
                    task_eps = df_episodes[df_episodes["strategy"] == task.name]
                    steps_path, eps_path = self.evaluator.save_trajectories(
                        task_steps, task_eps, task.name, self.meta_step
                    )
                    self.logger.log_artifact(
                        name=f"trajectories_{task.name}_step{self.meta_step}",
                        path=str(steps_path),
                    )
                    self.logger.log_artifact(
                        name=f"episodes_{task.name}_step{self.meta_step}",
                        path=str(eps_path),
                    )

        self.logger.close()

    def save_checkpoint(self, step: int) -> Path:
        """Save checkpoint to config checkpoint_dir / step.pt.

        Checkpoint contains: meta_params, outer_optimizer_state,
        meta_step, config, task_names.

        Returns:
            Path to saved checkpoint file.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.checkpoint_dir / f"{step}.pt"
        torch.save(
            {
                "meta_params": self.model.state_dict(),
                "outer_optimizer_state": self.fomaml.meta_optimizer.state_dict(),
                "meta_step": step,
                "config": self.config,
                "task_names": [t.name for t in self.tasks],
            },
            ckpt_path,
        )
        return ckpt_path

    def load_checkpoint(self, path: str | Path) -> int:
        """Load checkpoint and restore training state.

        Returns:
            meta_step: The step count at time of checkpoint.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["meta_params"])
        self.fomaml.meta_optimizer.load_state_dict(ckpt["outer_optimizer_state"])
        self.meta_step = ckpt["meta_step"]
        return self.meta_step
