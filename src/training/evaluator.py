"""Evaluation: fine-tune meta-policy per task, collect trajectories (vectorized)."""
import functools
import os
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.agents.actor_critic import ActorCritic
from src.agents.a2c import a2c_loss
from src.agents.ppo import ppo_loss
from src.env.yahtzee_env import env_reset, env_step, make_obs, get_action_mask
from src.env.constants import N_DICE, PHASE_ROLL, PHASE_SCORE, UPPER_BONUS_THRESHOLD
from src.meta.inner_loop import collect_episodes, prepare_ppo_data
from src.tasks.reward_tasks import TASK_NAMES


def _build_batched_eval(model, n_columns, max_steps, threshold=250):
    """Build a JIT-compiled function that runs N eval episodes in parallel.

    Returns a function: (params, rngs) -> (traj, episode_summary)
    where rngs has shape (n_episodes, 2) and all episodes run via vmap.
    """

    def _single_episode(params, rng):
        rng, init_rng = jax.random.split(rng)
        state, _ = env_reset(init_rng, n_columns)

        def scan_step(carry, _):
            state, rng, done = carry
            rng, act_rng = jax.random.split(rng)

            mask = get_action_mask(state, n_columns)
            obs = make_obs(state, n_columns)
            logits, value = model.apply(params, obs, state.phase)
            logits = jnp.where(mask, logits, -jnp.inf)
            probs = jax.nn.softmax(logits)
            action = jnp.argmax(logits)

            log_probs = jax.nn.log_softmax(logits)
            safe_lp = jnp.where(mask, log_probs, 0.0)
            entropy = -jnp.sum(probs * safe_lp)
            top3_idx = jnp.argsort(probs)[::-1][:3]
            top3_probs = probs[top3_idx]

            new_state, _, _, done_flag, _ = env_step(
                state, action, jnp.int32(0), n_columns, threshold)

            cat = action // n_columns
            col = action % n_columns

            step_out = {
                "phase": state.phase,
                "rerolls": state.rerolls,
                "dice": state.dice,
                "action": action,
                "value": value.squeeze(),
                "entropy": entropy,
                "top3_probs": top3_probs,
                "cum_score": jnp.sum(new_state.scores),
                "upper_total": jnp.sum(new_state.scores[:6, :]),
                "upper_bonus_achieved": jnp.any(
                    jnp.sum(new_state.scores[:6, :], axis=0)
                    >= UPPER_BONUS_THRESHOLD),
                "yb_triggered": new_state.yahtzee_bonus > state.yahtzee_bonus,
                "n_cols_completed": jnp.sum(
                    jnp.all(new_state.filled_mask, axis=0)),
                "score_gained": new_state.scores[cat, col],
                "cat": cat,
                "col": col,
                "done": done_flag,
                "valid": ~done,
            }

            next_state = jax.tree.map(
                lambda new, old: jnp.where(done, old, new),
                new_state, state)
            return (next_state, rng, done | done_flag), step_out

        (final_state, _, _), traj = jax.lax.scan(
            scan_step, (state, rng, jnp.bool_(False)), None, length=max_steps)

        # Episode-level summary from frozen final state
        fs = final_state.scores
        upper_per_col = jnp.sum(fs[:6, :], axis=0)
        ep_summary = {
            "final_score": jnp.sum(fs),
            "upper_score": jnp.sum(fs[:6, :]),
            "lower_score": jnp.sum(fs[6:, :]),
            "upper_bonus_total": jnp.sum(
                jnp.where(upper_per_col >= UPPER_BONUS_THRESHOLD, 35, 0)),
            "yahtzee_bonus_total": final_state.yahtzee_bonus,
            "n_yahtzees": jnp.sum(fs[11, :] == 50),
        }
        return traj, ep_summary

    @jax.jit
    def batched_eval(params, rngs):
        return jax.vmap(_single_episode, in_axes=(None, 0))(params, rngs)

    return batched_eval


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.n_columns = config["env"]["n_columns"]
        self.max_steps = config["env"]["max_steps_per_episode"]
        self.threshold = config["tasks"]["threshold_beater_score"]
        self.model = ActorCritic(
            hidden_dim=config["agent"]["hidden_dim"],
            n_layers=config["agent"]["n_layers"],
            n_columns=self.n_columns,
            use_layer_norm=config["agent"].get("use_layer_norm", False),
            activation=config["agent"].get("activation", "relu"),
        )
        self.inner_lr = config["meta"]["inner_lr"]
        self.n_inner_steps = config["meta"]["n_inner_steps"]
        if "a2c" in config:
            self.loss_fn = a2c_loss
            algo_config = config["a2c"]
        else:
            self.loss_fn = ppo_loss
            algo_config = config["ppo"]
        self.gamma = algo_config.get("gamma", 0.99)
        self.gae_lambda = algo_config.get("gae_lambda", 0.95)
        n_parallel = config["meta"].get("n_parallel_envs", 1)
        self.n_support_envs = max(n_parallel, 10)
        self._batched_eval = _build_batched_eval(
            self.model, self.n_columns, self.max_steps, self.threshold)

    def _adapt_params(self, meta_params, rng, task_id):
        """Fine-tune meta-params on one support rollout for a task."""
        support_traj = collect_episodes(
            meta_params, self.model, rng, task_id,
            self.n_support_envs, self.n_columns, self.max_steps,
            self.threshold)

        support_data = prepare_ppo_data(support_traj, gamma=self.gamma, lam=self.gae_lambda)
        obs_s, actions_s, phases_s, masks_s, lp_s, adv_s, ret_s = support_data

        fast_params = jax.tree.map(jnp.copy, meta_params)
        inner_lr = self.inner_lr

        for _ in range(self.n_inner_steps):
            loss, grads = jax.value_and_grad(self.loss_fn)(
                fast_params, self.model, obs_s, actions_s, phases_s,
                masks_s, lp_s, adv_s, ret_s)
            grads = jax.lax.stop_gradient(grads)
            grad_norm = jnp.sqrt(jax.tree_util.tree_reduce(
                lambda acc, g: acc + jnp.sum(g ** 2), grads, initializer=0.0))
            scale = jnp.minimum(1.0, 1.0 / (grad_norm + 1e-8))
            grads = jax.tree.map(lambda g: g * scale, grads)
            fast_params = jax.tree.map(
                lambda p, g: p - inner_lr * g, fast_params, grads)

        return fast_params

    def evaluate(self, params, meta_step, n_episodes=100):
        """Evaluate with per-task adaptation. Returns (df_steps, df_episodes).

        Vectorized: all episodes for a task run in a single JIT call via vmap.
        """
        all_step_rows = []
        all_ep_rows = []
        rng = jax.random.PRNGKey(meta_step)

        for task_id, task_name in enumerate(
                tqdm(TASK_NAMES, desc=f"Eval step {meta_step}", unit="task")):
            rng, adapt_rng, eval_rng = jax.random.split(rng, 3)
            adapted_params = self._adapt_params(
                params, adapt_rng, jnp.int32(task_id))

            ep_rngs = jax.random.split(eval_rng, n_episodes)
            traj, ep_summary = self._batched_eval(adapted_params, ep_rngs)

            # Transfer to CPU once
            traj_np = jax.device_get(traj)
            ep_np = jax.device_get(ep_summary)

            step_rows, ep_rows = _trajectories_to_dataframes(
                traj_np, ep_np, task_name, meta_step, n_episodes,
                self.n_columns, self.max_steps, self.threshold)
            all_step_rows.extend(step_rows)
            all_ep_rows.extend(ep_rows)

        df_steps = pd.DataFrame(all_step_rows)
        df_episodes = pd.DataFrame(all_ep_rows)
        return df_steps, df_episodes

    def save_trajectories(self, df_steps, df_episodes, strategy, meta_step, out_dir=None):
        out_dir = out_dir or self.config["training"].get("checkpoint_dir", "checkpoints")
        steps_path = os.path.join(
            out_dir, f"eval_steps_{strategy}_{meta_step}.parquet")
        episodes_path = os.path.join(
            out_dir, f"eval_episodes_{strategy}_{meta_step}.parquet")
        df_steps.to_parquet(steps_path, index=False)
        df_episodes.to_parquet(episodes_path, index=False)
        return steps_path, episodes_path


def _trajectories_to_dataframes(traj, ep_summary, task_name, meta_step,
                                n_episodes, n_columns, max_steps,
                                threshold=250):
    """Convert numpy trajectory arrays to row dicts for DataFrames."""
    step_rows = []
    ep_rows = []

    for ep in range(n_episodes):
        turn = 0
        round_in_turn = 0
        n_cross_outs = 0
        n_zeros = 0

        for t in range(max_steps):
            if not traj["valid"][ep, t]:
                break

            phase = int(traj["phase"][ep, t])
            row = {
                "episode_id": ep,
                "strategy": task_name,
                "meta_step": meta_step,
                "turn": turn,
                "round": round_in_turn,
                "phase": "roll" if phase == PHASE_ROLL else "score",
                "rerolls_remaining": int(traj["rerolls"][ep, t]),
            }
            for d in range(N_DICE):
                row[f"dice_{d+1}"] = int(traj["dice"][ep, t, d])

            if phase == PHASE_ROLL:
                row["dice_kept_mask"] = int(traj["action"][ep, t])
                row["action_category"] = None
                row["action_column"] = None
                row["score_gained"] = 0
                row["cross_out"] = False
                round_in_turn += 1
            else:
                sg = int(traj["score_gained"][ep, t])
                row["dice_kept_mask"] = None
                row["action_category"] = int(traj["cat"][ep, t])
                row["action_column"] = int(traj["col"][ep, t])
                row["score_gained"] = sg
                row["cross_out"] = sg == 0
                if sg == 0:
                    n_cross_outs += 1
                    n_zeros += 1
                turn += 1
                round_in_turn = 0

            row["cumulative_score"] = int(traj["cum_score"][ep, t])
            row["upper_section_total"] = int(traj["upper_total"][ep, t])
            row["upper_bonus_achieved"] = bool(traj["upper_bonus_achieved"][ep, t])
            row["value_estimate"] = float(traj["value"][ep, t])
            row["action_entropy"] = float(traj["entropy"][ep, t])
            row["action_probs_top3"] = [
                float(p) for p in traj["top3_probs"][ep, t]]
            row["yahtzee_bonus_triggered"] = bool(traj["yb_triggered"][ep, t])
            row["n_columns_completed"] = int(traj["n_cols_completed"][ep, t])
            step_rows.append(row)

        final_score = int(ep_summary["final_score"][ep])
        upper_score = int(ep_summary["upper_score"][ep])

        ep_rows.append({
            "episode_id": ep,
            "strategy": task_name,
            "meta_step": meta_step,
            "final_score": final_score,
            "upper_section_score": upper_score,
            "lower_section_score": final_score - upper_score,
            "upper_bonus_total": int(ep_summary["upper_bonus_total"][ep]),
            "yahtzee_bonus_total": int(ep_summary["yahtzee_bonus_total"][ep]),
            "n_yahtzees_scored": int(ep_summary["n_yahtzees"][ep]),
            "n_cross_outs": n_cross_outs,
            "n_zeros": n_zeros,
            "n_turns": turn,
            "beat_threshold": final_score >= threshold,
        })

    return step_rows, ep_rows
