"""Evaluation — fine-tune meta-policy per task, collect trajectories, save parquet."""

from pathlib import Path

import json
import numpy as np
import pandas as pd
import torch

from agents.actor_critic import ActorCritic
from agents.ppo import compute_gae
from env.constants import YAHTZEE
from env.yahtzee_env import YahtzeeEnv
from meta.inner_loop import clone_params, collect_episode, inner_update
from tasks.reward_tasks import TASK_REGISTRY


class Evaluator:
    """Evaluates a trained meta-policy checkpoint.

    For each task:
    1. Load meta-policy from checkpoint
    2. Fine-tune for M inner loop steps on the task
    3. Run N evaluation episodes via _collect_episode_eval
    4. Save trajectory and episode summary as parquet files

    Per-step trajectory schema (25 fields):
        episode_id, strategy, meta_step, turn, round, phase,
        rerolls_remaining, dice_1-5, dice_kept_mask,
        action_category, action_column, score_gained, cumulative_score,
        upper_section_total, upper_bonus_achieved,
        value_estimate, action_entropy, action_probs_top3,
        cross_out, yahtzee_bonus_triggered, n_columns_completed

    Per-episode summary schema (13 fields):
        episode_id, strategy, meta_step, final_score,
        upper_section_score, lower_section_score, upper_bonus_total,
        yahtzee_bonus_total, n_yahtzees_scored, n_cross_outs,
        n_zeros, n_turns, beat_threshold_250
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

    def _collect_episode_eval(
        self,
        env,
        fast_params: dict,
        meta_step: int,
        ep_idx: int,
        task_name: str,
    ) -> tuple[list[dict], dict]:
        """Run one episode with adapted params, capturing the full trajectory schema.

        Obs vector layout (from yahtzee_env.py):
            obs[0:5]                              dice values (1-6)
            obs[5]                                rerolls_remaining (0-2)
            obs[6 : 6 + 13*n_cols]               filled_mask (flattened, row-major)
            obs[6 + 13*n_cols : 6 + 26*n_cols]   scores (flattened, same order)
            obs[6 + 26*n_cols]                    yahtzee_bonus

        Args:
            env: YahtzeeEnv instance with reward_fn already set.
            fast_params: Task-adapted parameters from inner loop fine-tuning.
            meta_step: Checkpoint step (for labelling rows).
            ep_idx: Episode index within this evaluation run.
            task_name: Reward task name (e.g. "MaxScore").

        Returns:
            (step_rows, ep_row): List of per-step dicts and one episode summary dict.
        """
        n_cols = self.config["env"]["n_columns"]

        # Param swap: load fast_params, restore original after episode
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(fast_params)

        step_rows = []
        turn = 0
        round_num = 0  # increments only on score-phase steps
        terminal_obs = None

        try:
            obs, info = env.reset()

            while True:
                phase = info["phase"]
                mask = info["action_mask"]
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device)

                # Forward pass: logits + value; no grad needed for eval
                with torch.no_grad():
                    logits, value_t = self.model(obs_t.unsqueeze(0), phase)
                    logits = logits.squeeze(0)
                    value_est = value_t.squeeze().item()

                # Masked distribution for action sampling + entropy + top-3 probs
                masked_logits = logits.masked_fill(mask_t == 0, float("-inf"))
                dist = torch.distributions.Categorical(logits=masked_logits)
                idx = dist.sample()
                entropy = dist.entropy().item()

                k = min(3, int((mask_t > 0).sum().item()))
                top_vals, top_idxs = dist.probs.topk(k)
                action_probs_top3 = json.dumps([
                    {"action": int(i), "prob": round(float(v), 6)}
                    for i, v in zip(top_idxs.tolist(), top_vals.tolist())
                ])

                # Decode obs fields before step
                dice = [int(obs[i]) for i in range(5)]
                rerolls_remaining = int(obs[5])
                yahtzee_bonus_before = float(obs[6 + 26 * n_cols])

                # Decode and apply action
                if phase == "score":
                    flat = idx.item()
                    action_category = flat // n_cols
                    action_column = flat % n_cols
                    action = np.array([action_category, action_column])
                    dice_kept_mask = None
                else:
                    action = idx.item()
                    action_category = None
                    action_column = None
                    dice_kept_mask = int(action)

                next_obs, _, terminated, truncated, next_info = env.step(action)
                done = terminated or truncated

                # Decode next_obs for post-step state
                next_scores = next_obs[6 + 13 * n_cols: 6 + 26 * n_cols].reshape(n_cols, 13)
                next_filled = next_obs[6: 6 + 13 * n_cols].reshape(n_cols, 13)
                yahtzee_bonus_after = float(next_obs[6 + 26 * n_cols])

                # Derived state fields (computed from post-step obs)
                upper_scores_per_col = next_scores[:, :6].sum(axis=1)
                upper_bonus_earned = int((upper_scores_per_col >= 63).sum()) * 35
                cumulative_score = (
                    int(next_scores.sum()) + upper_bonus_earned + int(yahtzee_bonus_after)
                )
                upper_section_total = int(next_scores[:, :6].sum())
                upper_bonus_achieved = bool((upper_scores_per_col >= 63).any())
                n_columns_completed = int((next_filled.sum(axis=1) == 13).sum())

                # Phase-specific fields
                if phase == "score":
                    round_num += 1
                    # Score placed = value now in that slot (slot was 0 before)
                    score_gained = int(next_scores[action_column][action_category])
                    cross_out = bool(next_info.get("cross_out", False))
                    yahtzee_bonus_triggered = yahtzee_bonus_after > yahtzee_bonus_before
                else:
                    score_gained = 0
                    cross_out = False
                    yahtzee_bonus_triggered = False

                step_rows.append({
                    "episode_id": ep_idx,
                    "strategy": task_name,
                    "meta_step": meta_step,
                    "turn": turn,
                    "round": round_num,
                    "phase": phase,
                    "rerolls_remaining": rerolls_remaining,
                    "dice_1": dice[0],
                    "dice_2": dice[1],
                    "dice_3": dice[2],
                    "dice_4": dice[3],
                    "dice_5": dice[4],
                    "dice_kept_mask": dice_kept_mask,
                    "action_category": action_category,
                    "action_column": action_column,
                    "score_gained": score_gained,
                    "cumulative_score": cumulative_score,
                    "upper_section_total": upper_section_total,
                    "upper_bonus_achieved": upper_bonus_achieved,
                    "value_estimate": value_est,
                    "action_entropy": entropy,
                    "action_probs_top3": action_probs_top3,
                    "cross_out": cross_out,
                    "yahtzee_bonus_triggered": yahtzee_bonus_triggered,
                    "n_columns_completed": n_columns_completed,
                })

                turn += 1
                terminal_obs = next_obs
                if done:
                    break
                obs = next_obs
                info = next_info

        finally:
            self.model.load_state_dict(original_state)

        # Episode summary — derived from terminal obs + step_rows
        final_scores = terminal_obs[6 + 13 * n_cols: 6 + 26 * n_cols].reshape(n_cols, 13)
        upper_per_col = final_scores[:, :6].sum(axis=1)
        upper_bonus_total_final = int((upper_per_col >= 63).sum()) * 35
        yahtzee_bonus_final = int(terminal_obs[6 + 26 * n_cols])
        final_score = int(final_scores.sum()) + upper_bonus_total_final + yahtzee_bonus_final

        score_steps = [r for r in step_rows if r["phase"] == "score"]
        n_cross_outs = sum(1 for r in score_steps if r["cross_out"])
        # YAHTZEE is category index 11 (src/env/constants.py); a Yahtzee scores exactly 50
        n_yahtzees_scored = sum(
            1 for r in score_steps
            if r["action_category"] == YAHTZEE and r["score_gained"] == 50
        )
        n_zeros = sum(1 for r in score_steps if r["score_gained"] == 0)

        ep_row = {
            "episode_id": ep_idx,
            "strategy": task_name,
            "meta_step": meta_step,
            "final_score": final_score,
            "upper_section_score": int(final_scores[:, :6].sum()),
            "lower_section_score": int(final_scores[:, 6:].sum()),
            "upper_bonus_total": upper_bonus_total_final,
            "yahtzee_bonus_total": yahtzee_bonus_final,
            "n_yahtzees_scored": n_yahtzees_scored,
            "n_cross_outs": n_cross_outs,
            "n_zeros": n_zeros,
            "n_turns": turn,
            "beat_threshold_250": bool(final_score > 250),
        }

        return step_rows, ep_row

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
                step_rows, ep_row = self._collect_episode_eval(
                    env=env2,
                    fast_params=fast_params,
                    meta_step=meta_step,
                    ep_idx=ep_idx,
                    task_name=task.name,
                )
                all_steps.extend(step_rows)
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
