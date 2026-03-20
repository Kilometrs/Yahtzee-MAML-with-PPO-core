"""PPO inner loop optimizer for FOMAML."""

import torch
from agents.actor_critic import ActorCritic
from agents.rollout_buffer import RolloutBuffer


class PPO:
    """Proximal Policy Optimization inner loop.

    Used as the task-specific optimizer within FOMAML. Can operate on either
    the original model parameters or on fast_params (cloned params for inner loop).
    """

    def __init__(
        self,
        actor_critic: ActorCritic,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        gae_lambda: float = 0.95,
        n_epochs: int = 4,
        batch_size: int = 64,
    ):
        self.actor_critic = actor_critic
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def compute_gae(self, buffer: RolloutBuffer, gamma: float = 0.99) -> torch.Tensor:
        """Compute Generalized Advantage Estimation.

        Args:
            buffer: Filled rollout buffer.
            gamma: Discount factor.

        Returns:
            advantages: Tensor of shape (T,).
        """
        pass

    def update(
        self,
        buffer: RolloutBuffer,
        fast_params: dict | None = None,
        inner_lr: float | None = None,
    ) -> float:
        """Run PPO update steps.

        If fast_params is provided, uses torch.func.functional_call for FOMAML
        inner loop (stateless update, returns updated fast_params).
        Otherwise updates actor_critic parameters in-place (standard training).

        Args:
            buffer: Filled rollout buffer with episode experience.
            fast_params: Optional cloned parameter dict for FOMAML inner loop.
            inner_lr: Learning rate for inner loop update (required if fast_params given).

        Returns:
            mean_loss: Average policy loss over update epochs.
        """
        pass
