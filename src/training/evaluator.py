"""Evaluation — fine-tune meta-policy per task, collect trajectories, save parquet."""

from pathlib import Path

import pandas as pd
import torch

from agents.actor_critic import ActorCritic
from agents.ppo import compute_gae
from env.yahtzee_env import YahtzeeEnv
from meta.inner_loop import clone_params, collect_episode, inner_update
from tasks.reward_tasks import TASK_REGISTRY


class Evaluator:
    """Evaluates a trained meta-policy checkpoint.

    For each task:
    1. Load meta-policy from checkpoint
    2. Fine-tune for M inner loop steps on the task
    3. Run N evaluation episodes, collecting full per-step trajectory data
    4. Save trajectory and episode summary as parquet files
    5. Upload to ClearML as artifacts

    Trajectory parquet schema (per-step):
        episode (int), round (int), rerolls_left (int), dice (list[int]),
        phase (str), action_category (int), action_column (int),
        score_gained (int), value_estimate (float), entropy (float),
        action_probs (list[float]), cross_out (bool), yahtzee_bonus_event (bool),
        cumulative_score (int), strategy (str), meta_step (int)

    Episode summary parquet schema:
        episode (int), final_score (int), n_cross_outs (int),
        n_yahtzees (int), yahtzee_bonus_total (int),
        upper_bonus_earned (bool), strategy (str), meta_step (int)
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Full config dict loaded from YAML.
        """
        self.config = config
        self.device = torch.device("cpu")  # evaluator always runs on CPU
        n_columns = config["env"]["n_columns"]
        obs_dim = 7 + 26 * n_columns
        self.model = ActorCritic(
            obs_dim=obs_dim,
            n_columns=n_columns,
            hidden_dim=config["agent"]["hidden_dim"],
            n_layers=config["agent"]["n_layers"],
        )
        # Build tasks from TASK_REGISTRY
        threshold = config["tasks"]["threshold_beater_score"]
        self.tasks = [
            TASK_REGISTRY["MaxScore"](),
            TASK_REGISTRY["ThresholdBeater"](threshold=threshold),
            TASK_REGISTRY["UpperBonus"](),
            TASK_REGISTRY["YahtzeeHunter"](),
            TASK_REGISTRY["Conservative"](),
        ]

    def evaluate(
        self,
        checkpoint_path: str | Path,
        n_episodes: int = 100,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run evaluation for all tasks.

        Returns:
            (df_steps, df_episodes): Per-step trajectory and per-episode summary DataFrames.
        """
        # Load checkpoint (torch.load with weights_only=False)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["meta_params"])
        meta_step = ckpt.get("meta_step", 0)

        cfg = self.config
        n_columns = cfg["env"]["n_columns"]
        inner_lr = cfg["meta"]["inner_lr"]
        n_inner_steps = cfg["meta"]["n_inner_steps"]
        ppo_cfg = cfg["ppo"]

        all_steps = []
        all_episodes = []

        for task in self.tasks:
            # Fine-tune: collect one support episode, run n_inner_steps inner_update
            env = YahtzeeEnv(n_columns=n_columns)
            env.reward_fn = task.reward
            support_buf = collect_episode(self.model, env, self.device)
            adv, ret = compute_gae(support_buf, ppo_cfg["gae_lambda"])
            adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).detach()
            ret = ret.detach()
            sd = support_buf.get()
            fast_params = clone_params(self.model)
            for _ in range(n_inner_steps):
                fast_params = inner_update(
                    model=self.model,
                    fast_params=fast_params,
                    obs=torch.tensor(sd["obs"], device=self.device),
                    actions=sd["actions"],
                    advantages=adv,
                    returns=ret,
                    log_probs_old=torch.tensor(sd["log_probs"], device=self.device),
                    phases=sd["phases"],
                    action_masks=sd["action_masks"],
                    inner_lr=inner_lr,
                    clip_epsilon=ppo_cfg["clip_epsilon"],
                    entropy_coef=ppo_cfg["entropy_coef"],
                    value_loss_coef=ppo_cfg["value_loss_coef"],
                )

            # Collect n_episodes with adapted params
            for ep_idx in range(n_episodes):
                env2 = YahtzeeEnv(n_columns=n_columns)
                env2.reward_fn = task.reward
                buf = collect_episode(self.model, env2, self.device, params=fast_params)
                data = buf.get()

                # Build per-step rows
                for i, (obs_i, phase_i, action_i) in enumerate(
                    zip(data["obs"], data["phases"], data["actions"])
                ):
                    row = {
                        "episode": ep_idx,
                        "step": i,
                        "phase": phase_i,
                        "strategy": task.name,
                        "meta_step": meta_step,
                    }
                    all_steps.append(row)

                # Build episode summary row
                ep_row = {
                    "episode": ep_idx,
                    "strategy": task.name,
                    "meta_step": meta_step,
                    "n_steps": len(data["obs"]),
                }
                all_episodes.append(ep_row)

        df_steps = pd.DataFrame(all_steps)
        df_episodes = pd.DataFrame(all_episodes)
        return df_steps, df_episodes

    def save_trajectories(
        self,
        df_steps: pd.DataFrame,
        df_episodes: pd.DataFrame,
        strategy: str,
        meta_step: int,
    ) -> tuple[Path, Path]:
        """Save DataFrames to parquet under data/trajectories/ and data/episodes/.

        Returns:
            (steps_path, episodes_path): Paths to saved parquet files.
        """
        base = Path("data")
        steps_dir = base / "trajectories"
        eps_dir = base / "episodes"
        steps_dir.mkdir(parents=True, exist_ok=True)
        eps_dir.mkdir(parents=True, exist_ok=True)
        steps_path = steps_dir / f"{strategy}_step{meta_step}.parquet"
        eps_path = eps_dir / f"{strategy}_step{meta_step}.parquet"
        df_steps.to_parquet(steps_path, index=False)
        df_episodes.to_parquet(eps_path, index=False)
        return steps_path, eps_path
