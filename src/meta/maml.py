"""FOMAML outer loop — meta-training across reward-shaped tasks."""

import torch
import torch.nn as nn
from meta.inner_loop import clone_params, inner_update


class FOMAML:
    """First-Order Model-Agnostic Meta-Learning outer loop.

    Meta-trains a policy initialization that can quickly adapt to any
    reward-shaped Yahtzee task via a few PPO inner loop steps.

    Outer loop (one meta-step):
        1. Sample K tasks from task distribution
        2. For each task:
           a. fast_params = clone_params(meta_policy)
           b. Collect rollout using fast_params on env with task reward
           c. Run n_inner_steps of inner_update on fast_params
        3. Compute meta-loss: average policy loss evaluated at adapted fast_params
        4. meta_optimizer.step() on meta-loss gradients w.r.t. original params
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

    def meta_update(self, tasks: list, env_fn) -> float:
        """Run one outer loop meta-update across a batch of tasks.

        Args:
            tasks: List of BaseTask instances sampled for this meta-step.
            env_fn: Callable() -> YahtzeeEnv (fresh env per task).

        Returns:
            meta_loss: Scalar meta-loss value for logging.
        """
        pass
