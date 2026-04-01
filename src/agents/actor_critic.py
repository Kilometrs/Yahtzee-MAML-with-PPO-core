"""Actor-Critic network for multi-column Yahtzee.

Architecture:
  - Shared trunk: MLP over flat observation
  - Roll head: outputs logits over Discrete(32) — which dice to keep
  - Score head: outputs logits over (13, n_columns) — which (category, column) to score
  - Value head: outputs scalar state value estimate

The active policy head is selected externally based on the current phase
('roll' or 'score'), determined by _phase in the environment.
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class ActorCritic(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        n_columns: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ):
        """
        Args:
            obs_dim: Flat observation vector size (7 + 26 * n_columns).
            n_columns: Number of scorecard columns.
            hidden_dim: Hidden layer width.
            n_layers: Number of hidden layers in shared trunk.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.n_columns = n_columns

        # Shared trunk
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # Policy heads
        self.roll_head = nn.Linear(hidden_dim, 32)              # Discrete(32)
        self.score_head = nn.Linear(hidden_dim, 13 * n_columns)  # MultiDiscrete([13, n_columns])

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: Tensor, phase: str) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            obs: Observation tensor of shape (batch, obs_dim).
            phase: "roll" or "score".

        Returns:
            (action_logits, value): logits shape depends on phase;
                roll -> (batch, 32), score -> (batch, 13 * n_columns).
                value shape: (batch, 1).
        """
        features = self.trunk(obs)
        logits = self.roll_head(features) if phase == "roll" else self.score_head(features)
        value = self.value_head(features)
        return logits, value

    def get_action(
        self,
        obs: Tensor,
        phase: str,
        action_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Sample an action and return (action, log_prob).

        Handles single observations (1-D) by unsqueezing internally.

        Args:
            obs: Observation tensor (obs_dim,) or (batch, obs_dim).
            phase: "roll" or "score".
            action_mask: Binary mask of valid actions. 1 = valid, 0 = invalid.

        Returns:
            (action, log_prob):
                roll phase  -> action is int, log_prob is scalar Tensor
                score phase -> action is np.array([category, column]), log_prob is scalar Tensor
        """
        squeeze = obs.dim() == 1
        if squeeze:
            obs = obs.unsqueeze(0)

        logits, _ = self.forward(obs, phase)

        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits.masked_fill(action_mask == 0, float("-inf"))

        dist = torch.distributions.Categorical(logits=logits)
        idx = dist.sample()
        log_prob = dist.log_prob(idx)

        if squeeze:
            idx = idx.squeeze(0)
            log_prob = log_prob.squeeze(0)

        if phase == "score":
            category = (idx // self.n_columns).item()
            column = (idx % self.n_columns).item()
            action = np.array([category, column])
        else:
            action = idx.item()

        return action, log_prob

    def get_value(self, obs: Tensor) -> Tensor:
        """Return critic value estimate.

        Args:
            obs: Observation tensor (obs_dim,) or (batch, obs_dim).

        Returns:
            value: Tensor of shape (batch, 1) or (1, 1) for single obs.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        features = self.trunk(obs)
        return self.value_head(features)
