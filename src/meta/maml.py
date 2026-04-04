"""FOMAML outer loop — meta-training across reward-shaped tasks."""

import torch
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Callable



from meta.inner_loop import (
    clone_params, inner_update, collect_episode, collect_episode_cpu,
    _ppo_loss_fn, _ppo_loss_fn_fast, inner_update_fast, preprocess_rollout_for_device,
)
from agents.ppo import compute_gae_from_arrays
from agents.rollout_buffer import RolloutBuffer


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
        n_workers: int = 4,
    ):
        """
        Args:
            model: The meta-policy ActorCritic (shared across all tasks).
            inner_lr: Learning rate for inner loop parameter updates.
            outer_lr: Learning rate for outer loop (Adam) meta-optimizer.
            n_inner_steps: Number of PPO steps per task in the inner loop.
            n_workers: Worker processes for parallel episode collection.
        """
        self.model = model
        self.inner_lr = inner_lr
        self.n_inner_steps = n_inner_steps
        self.n_workers = n_workers
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
        if n_workers > 1:
            self._pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=get_context("spawn"))
        else:
            self._pool = None

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

        # Snapshot of meta-params on CPU — shared across all worker calls
        cpu_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        model_kwargs = {
            "obs_dim": self.model.obs_dim,
            "n_columns": self.model.n_columns,
            "hidden_dim": self.model.trunk[0].out_features,
            "n_layers": len([m for m in self.model.trunk if isinstance(m, torch.nn.Linear)]),
        }

        n_columns = self.model.n_columns

        # --- Parallel support rollout collection (CPU workers) ---
        if self._pool is not None:
            support_futs = [
                self._pool.submit(collect_episode_cpu, cpu_state, model_kwargs, task, n_columns)
                for task in tasks
            ]
            support_data = [f.result() for f in support_futs]
        else:
            support_data = [
                collect_episode_cpu(cpu_state, model_kwargs, task, n_columns)
                for task in tasks
            ]

        # --- Inner loop adaptation + pipelined query collection ---
        # As each task's inner loop finishes on GPU, immediately submit
        # its query collection to CPU workers (overlaps GPU and CPU work).
        fast_params_list = []
        query_futs = []
        for task, sd in zip(tasks, support_data):
            adv_s, ret_s = compute_gae_from_arrays(
                sd["rewards"], sd["values"], sd["dones"], gae_lambda,
            )
            adv_s = ((adv_s - adv_s.mean()) / (adv_s.std() + 1e-8)).detach().to(device)
            ret_s = ((ret_s - ret_s.mean()) / (ret_s.std() + 1e-8)).detach().to(device)

            # Pre-process once — avoids repeated numpy→tensor conversions in inner loop
            preprocessed = preprocess_rollout_for_device(sd, device, n_columns)

            fast_params = clone_params(self.model)
            for _ in range(self.n_inner_steps):
                fast_params = inner_update_fast(
                    model=self.model,
                    fast_params=fast_params,
                    preprocessed=preprocessed,
                    advantages=adv_s,
                    returns=ret_s,
                    inner_lr=self.inner_lr,
                    clip_epsilon=clip_epsilon,
                    entropy_coef=entropy_coef,
                    value_loss_coef=value_loss_coef,
                )
            fast_params_list.append(fast_params)

            # Submit query collection immediately (overlaps with next task's inner loop)
            fp_cpu = {k: v.detach().cpu() for k, v in fast_params.items()}
            if self._pool is not None:
                query_futs.append(
                    self._pool.submit(collect_episode_cpu, cpu_state, model_kwargs,
                                      task, n_columns, fp_cpu)
                )
            else:
                query_futs.append(
                    collect_episode_cpu(cpu_state, model_kwargs, task, n_columns, fp_cpu)
                )

        # Collect query results (most should already be done due to pipelining)
        if self._pool is not None:
            query_data = [f.result() for f in query_futs]
        else:
            query_data = query_futs

        # --- Meta-gradient computation (GPU) ---
        for task, fast_params, qd in zip(tasks, fast_params_list, query_data):
            adv_q, ret_q = compute_gae_from_arrays(
                qd["rewards"], qd["values"], qd["dones"], gae_lambda,
            )
            adv_q = ((adv_q - adv_q.mean()) / (adv_q.std() + 1e-8)).detach().to(device)
            ret_q = ((ret_q - ret_q.mean()) / (ret_q.std() + 1e-8)).detach().to(device)

            q_preprocessed = preprocess_rollout_for_device(qd, device, n_columns)
            query_loss = _ppo_loss_fn_fast(
                model=self.model,
                fast_params=fast_params,
                preprocessed=q_preprocessed,
                advantages=adv_q,
                returns=ret_q,
                clip_epsilon=clip_epsilon,
                entropy_coef=entropy_coef,
                value_loss_coef=value_loss_coef,
            )

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
