"""FOMAML outer loop — meta-training across reward-shaped tasks."""

import torch
import torch.nn as nn
from typing import Callable

from meta.inner_loop import clone_params, inner_update, collect_episode, _ppo_loss_fn
from agents.ppo import compute_gae


class FOMAML:
    """First-Order Model-Agnostic Meta-Learning outer loop.

    Meta-trains a policy initialization that can quickly adapt to any
    reward-shaped Yahtzee task via a few PPO inner loop steps.

    Outer loop (one meta-step):
        1. Sample K tasks from task distribution (done by MetaTrainer)
        2. For each task:
           a. Collect support rollout with meta-params
           b. fast_params = clone_params(meta_policy)
           c. n_inner_steps of inner_update on support data
           d. Collect query rollout with adapted fast_params (param-swap)
           e. Compute query loss at fast_params via functional_call
           f. Accumulate task_grads / K onto meta_params.grad
        3. meta_optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float,
        outer_lr: float,
        n_inner_steps: int,
    ):
        """
        Args:
            model: The meta-policy ActorCritic (shared across all tasks).
            inner_lr: Learning rate for inner loop parameter updates.
            outer_lr: Learning rate for outer loop (Adam) meta-optimizer.
            n_inner_steps: Number of PPO steps per task in the inner loop.
        """
        self.model = model
        self.inner_lr = inner_lr
        self.n_inner_steps = n_inner_steps
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)

    def meta_update(
        self,
        tasks: list,
        env_fn: Callable,
        ppo_cfg: dict,
        device: torch.device,
    ) -> tuple[float, dict[str, float]]:
        """Run one outer loop meta-update across a batch of tasks.

        For each task:
        1. Collect support rollout with current meta-params
        2. Run n_inner_steps of inner_update to get adapted fast_params
        3. Collect query rollout using adapted fast_params (param-swap)
        4. Compute query loss at fast_params via functional_call
        5. Accumulate meta-gradient (dL_query/d_fast_params) onto meta_params.grad

        This is FOMAML: the meta-gradient is the first-order approximation
        (gradient at adapted params, treating them as if they were the meta-params).

        Args:
            tasks: List of BaseTask instances sampled for this meta-step.
            env_fn: Callable() -> YahtzeeEnv (fresh env per task).
            ppo_cfg: Dict with keys clip_epsilon, entropy_coef,
                     value_loss_coef, gae_lambda.
            device: Torch device.

        Returns:
            (meta_loss, per_task_losses): Mean query loss and dict of task_name -> loss.
        """
        self.meta_optimizer.zero_grad()
        total_query_loss = 0.0
        per_task_losses: dict[str, float] = {}
        gae_lambda = ppo_cfg["gae_lambda"]
        clip_epsilon = ppo_cfg["clip_epsilon"]
        entropy_coef = ppo_cfg["entropy_coef"]
        value_loss_coef = ppo_cfg["value_loss_coef"]

        for task in tasks:
            env = env_fn()
            env.reward_fn = task.reward  # inject task-specific reward

            # --- Support rollout (meta-params) ---
            support_buf = collect_episode(self.model, env, device)
            adv_s, ret_s = compute_gae(support_buf, gae_lambda)
            adv_s = ((adv_s - adv_s.mean()) / (adv_s.std() + 1e-8)).detach().to(device)
            ret_s = ((ret_s - ret_s.mean()) / (ret_s.std() + 1e-8)).detach().to(device)
            sd = support_buf.get()

            # --- Inner loop adaptation ---
            fast_params = clone_params(self.model)
            for _ in range(self.n_inner_steps):
                fast_params = inner_update(
                    model=self.model,
                    fast_params=fast_params,
                    obs=torch.tensor(sd["obs"], device=device),
                    actions=sd["actions"],
                    advantages=adv_s,
                    returns=ret_s,
                    log_probs_old=torch.tensor(sd["log_probs"], device=device),
                    phases=sd["phases"],
                    action_masks=sd["action_masks"],
                    inner_lr=self.inner_lr,
                    clip_epsilon=clip_epsilon,
                    entropy_coef=entropy_coef,
                    value_loss_coef=value_loss_coef,
                )

            # --- Query rollout (adapted fast_params via param-swap) ---
            query_buf = collect_episode(self.model, env, device, params=fast_params)
            adv_q, ret_q = compute_gae(query_buf, gae_lambda)
            adv_q = ((adv_q - adv_q.mean()) / (adv_q.std() + 1e-8)).detach().to(device)
            ret_q = ((ret_q - ret_q.mean()) / (ret_q.std() + 1e-8)).detach().to(device)
            qd = query_buf.get()

            # --- Meta-gradient: query loss at adapted params ---
            query_loss = _ppo_loss_fn(
                model=self.model,
                fast_params=fast_params,
                obs=torch.tensor(qd["obs"], device=device),
                actions=qd["actions"],
                advantages=adv_q,
                returns=ret_q,
                log_probs_old=torch.tensor(qd["log_probs"], device=device),
                phases=qd["phases"],
                action_masks=qd["action_masks"],
                clip_epsilon=clip_epsilon,
                entropy_coef=entropy_coef,
                value_loss_coef=value_loss_coef,
            )

            # FOMAML: treat gradient at adapted params as meta-gradient
            # (first-order approximation — no differentiation through inner update)
            # Guard: _ppo_loss_fn returns a leaf tensor with no grad_fn when
            # total_count == 0 (empty rollout). In that case, skip gradient
            # accumulation for this task; it still counts in len(tasks) so the
            # meta-gradient is diluted — but empty Yahtzee episodes cannot occur
            # in practice (every game produces at least one scoring step).
            if query_loss.grad_fn is not None:
                task_grads = torch.autograd.grad(
                    query_loss,
                    list(fast_params.values()),
                    allow_unused=True,
                )
                task_grads = [
                    g if g is not None else torch.zeros_like(v)
                    for g, v in zip(task_grads, fast_params.values())
                ]

                # Accumulate onto meta_params.grad (divided by task count)
                for param, grad in zip(self.model.parameters(), task_grads):
                    if param.grad is None:
                        param.grad = grad.detach().clone() / len(tasks)
                    else:
                        param.grad += grad.detach().clone() / len(tasks)

            per_task_losses[task.name] = query_loss.item()
            total_query_loss += query_loss.item()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Skip update if any gradient is NaN or Inf (numerical instability guard)
        has_bad_grad = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in self.model.parameters()
        )
        if not has_bad_grad:
            self.meta_optimizer.step()
        else:
            self.meta_optimizer.zero_grad()

        return total_query_loss / len(tasks), per_task_losses
