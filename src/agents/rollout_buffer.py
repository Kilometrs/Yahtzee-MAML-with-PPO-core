"""Rollout buffer for storing PPO experience."""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class RolloutBuffer:
    """Stores one episode's worth of transition data for PPO.

    Two fields beyond the original spec are added:
      phases:       "roll" or "score" per step — needed by PPO to select the
                    correct actor-critic head when recomputing log-probs.
      action_masks: the action mask at each step — needed to rebuild masked
                    distributions during the PPO update.
    """

    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)
    phases: list = field(default_factory=list)
    action_masks: list = field(default_factory=list)

    def add(
        self,
        obs: np.ndarray,
        action,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        phase: str,
        action_mask: np.ndarray,
    ) -> None:
        """Append one transition to the buffer."""
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.phases.append(phase)
        self.action_masks.append(action_mask)

    def clear(self) -> None:
        """Reset buffer to empty state."""
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.phases = []
        self.action_masks = []

    def get(self) -> dict:
        """Return buffer contents as arrays.

        Returns:
            Dict with keys:
              obs, rewards, dones, log_probs, values  -> float32 numpy arrays
              actions, phases, action_masks           -> object numpy arrays
        """
        return {
            "obs": np.array(self.obs, dtype=np.float32),
            "actions": np.array(self.actions, dtype=object),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "dones": np.array(self.dones, dtype=np.float32),
            "log_probs": np.array(self.log_probs, dtype=np.float32),
            "values": np.array(self.values, dtype=np.float32),
            "phases": np.array(self.phases, dtype=object),
            "action_masks": np.array(self.action_masks, dtype=object),
        }

    def __len__(self) -> int:
        return len(self.rewards)
