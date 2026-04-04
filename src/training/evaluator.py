"""Evaluation: fine-tune meta-policy per task, collect trajectories."""
import os
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from src.agents.actor_critic import ActorCritic
from src.env.yahtzee_env import env_reset, env_step, make_obs, get_action_mask
from src.env.constants import N_DICE, PHASE_ROLL, PHASE_SCORE, UPPER_BONUS_THRESHOLD
from src.meta.inner_loop import collect_episodes, prepare_ppo_data, inner_update_and_query_grad
from src.tasks.reward_tasks import TASK_NAMES


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.n_columns = config["env"]["n_columns"]
        self.max_steps = config["env"]["max_steps_per_episode"]
        self.model = ActorCritic(
            hidden_dim=config["agent"]["hidden_dim"],
            n_layers=config["agent"]["n_layers"],
            n_columns=self.n_columns)

    def _collect_eval_episode(self, params, rng, task_name, meta_step, ep_idx):
        """Collect one episode with detailed trajectory recording (Python loop)."""
        n_cols = self.n_columns
        state, obs = env_reset(rng, n_cols)
        step_rows = []
        turn = 0
        round_in_turn = 0
        done = False

        while not done:
            phase = int(state.phase)
            mask = get_action_mask(state, n_cols)
            obs_arr = make_obs(state, n_cols)
            logits, value = self.model.apply(params, obs_arr, state.phase)
            logits = jnp.where(mask, logits, -jnp.inf)
            probs = jax.nn.softmax(logits)
            action = jnp.argmax(logits)
            log_probs = jax.nn.log_softmax(logits)
            safe_log_probs = jnp.where(mask, log_probs, 0.0)
            entropy = float(-jnp.sum(probs * safe_log_probs))
            top3_idx = jnp.argsort(probs)[::-1][:3]
            top3_probs = probs[top3_idx]

            rng, step_rng = jax.random.split(rng)
            new_state, new_obs, reward, done_flag, info = env_step(
                state, action, jnp.int32(0), n_cols)
            done = bool(done_flag)

            row = {
                "episode_id": ep_idx,
                "strategy": task_name,
                "meta_step": meta_step,
                "turn": turn,
                "round": round_in_turn,
                "phase": "roll" if phase == PHASE_ROLL else "score",
                "rerolls_remaining": int(state.rerolls),
            }
            for d in range(N_DICE):
                row[f"dice_{d+1}"] = int(state.dice[d])
            if phase == PHASE_ROLL:
                row["dice_kept_mask"] = int(action)
                row["action_category"] = None
                row["action_column"] = None
                row["score_gained"] = 0
                row["cross_out"] = False
                round_in_turn += 1
            else:
                cat = int(action) // n_cols
                col = int(action) % n_cols
                score_gained = int(new_state.scores[cat, col])
                row["dice_kept_mask"] = None
                row["action_category"] = cat
                row["action_column"] = col
                row["score_gained"] = score_gained
                row["cross_out"] = score_gained == 0
                turn += 1
                round_in_turn = 0

            row["cumulative_score"] = int(jnp.sum(new_state.scores))
            row["upper_section_total"] = int(jnp.sum(new_state.scores[:6, :]))
            row["upper_bonus_achieved"] = bool(
                jnp.any(jnp.sum(new_state.scores[:6, :], axis=0) >= UPPER_BONUS_THRESHOLD))
            row["value_estimate"] = float(value)
            row["action_entropy"] = entropy
            row["action_probs_top3"] = [float(p) for p in top3_probs]
            row["yahtzee_bonus_triggered"] = int(new_state.yahtzee_bonus) > int(state.yahtzee_bonus)
            row["n_columns_completed"] = int(jnp.sum(jnp.all(new_state.filled_mask, axis=0)))
            step_rows.append(row)
            state = new_state

        final_score = int(jnp.sum(state.scores))
        upper_score = int(jnp.sum(state.scores[:6, :]))
        lower_score = int(jnp.sum(state.scores[6:, :]))
        ep_row = {
            "episode_id": ep_idx,
            "strategy": task_name,
            "meta_step": meta_step,
            "final_score": final_score,
            "upper_section_score": upper_score,
            "lower_section_score": lower_score,
            "upper_bonus_total": int(jnp.sum(
                jnp.where(jnp.sum(state.scores[:6, :], axis=0) >= UPPER_BONUS_THRESHOLD, 35, 0))),
            "yahtzee_bonus_total": int(state.yahtzee_bonus),
            "n_yahtzees_scored": int(jnp.sum(state.scores[11, :] == 50)),
            "n_cross_outs": sum(1 for r in step_rows if r.get("cross_out", False)),
            "n_zeros": sum(1 for r in step_rows if r.get("score_gained", -1) == 0 and r["phase"] == "score"),
            "n_turns": turn,
            "beat_threshold_250": final_score >= 250,
        }
        return step_rows, ep_row

    def evaluate(self, params, meta_step, n_episodes=100):
        """Evaluate with meta params. Returns (df_steps, df_episodes)."""
        all_step_rows = []
        all_ep_rows = []
        rng = jax.random.PRNGKey(meta_step)

        for task_id, task_name in enumerate(TASK_NAMES):
            for ep_idx in range(n_episodes):
                rng, ep_rng = jax.random.split(rng)
                step_rows, ep_row = self._collect_eval_episode(
                    params, ep_rng, task_name, meta_step, ep_idx)
                all_step_rows.extend(step_rows)
                all_ep_rows.append(ep_row)

        df_steps = pd.DataFrame(all_step_rows)
        df_episodes = pd.DataFrame(all_ep_rows)
        return df_steps, df_episodes

    def save_trajectories(self, df_steps, df_episodes, strategy, meta_step):
        out_dir = self.config["training"]["checkpoint_dir"]
        steps_path = os.path.join(out_dir, f"eval_steps_{strategy}_{meta_step}.parquet")
        episodes_path = os.path.join(out_dir, f"eval_episodes_{strategy}_{meta_step}.parquet")
        df_steps.to_parquet(steps_path, index=False)
        df_episodes.to_parquet(episodes_path, index=False)
        return steps_path, episodes_path
