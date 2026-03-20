"""Rollout buffer for storing PPO experience."""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class RolloutBuffer:
    """Stores one episode's worth of (obs, action, reward, done, log_prob, value) tuples.

    Used by PPO inner loop for computing GAE advantages and policy loss.
    """

    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)

    def add(
        self,
        obs: np.ndarray,
        action,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Append one transition to the buffer."""
        pass

    def clear(self) -> None:
        """Reset buffer to empty state."""
        pass

    def get(self) -> dict:
        """Return buffer contents as numpy arrays.

        Returns:
            Dict with keys: obs, actions, rewards, dones, log_probs, values.
        """
        pass

    def __len__(self) -> int:
        return len(self.rewards)
