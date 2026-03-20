"""Multi-column Yahtzee Gymnasium environment."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.constants import N_DICE, N_CATEGORIES, MAX_REROLLS


class YahtzeeEnv(gym.Env):
    """Single-player multi-column Yahtzee environment.

    Observation space (flat float32 vector):
        - dice values (5,): raw integers 1-6
        - remaining_rerolls (1,): integer 0-2
        - filled_mask (13 * n_columns,): binary 0/1 per slot
        - current_scores (13 * n_columns,): raw integers (0 if unfilled)
        - yahtzee_bonus (1,): global bonus counter, multiples of 100

    Total obs size: 7 + 26 * n_columns

    Action space:
        Roll phase:  Discrete(32) — bitmask of dice to keep (0-31).
                     Value 31 (all kept) ends roll phase immediately.
        Score phase: MultiDiscrete([13, n_columns]) — (category, column).

    The current phase is determined by remaining_rerolls in the observation.
    Action masks are returned in info["action_mask"].
    """

    metadata = {"render_modes": []}

    def __init__(self, n_columns: int = 3, reward_fn=None):
        """
        Args:
            n_columns: Number of scorecard columns (default 3 for dev, 6 for full).
            reward_fn: Optional callable(state, action, next_state, done, info) -> float.
                       If None, uses raw score delta as reward.
        """
        super().__init__()
        self.n_columns = n_columns
        self.reward_fn = reward_fn

        obs_size = 7 + 26 * n_columns
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Two action spaces; active one depends on phase
        self.roll_action_space = spaces.Discrete(32)
        self.score_action_space = spaces.MultiDiscrete([N_CATEGORIES, n_columns])

        # Internal state (populated on reset)
        self._dice = None
        self._remaining_rerolls = None
        self._filled_mask = None   # shape: (n_columns, N_CATEGORIES)
        self._scores = None        # shape: (n_columns, N_CATEGORIES)
        self._yahtzee_bonus = None
        self._phase = None         # "roll" or "score"

    def reset(self, seed=None, options=None):
        """Reset environment to a new game.

        Returns:
            obs: Initial observation vector.
            info: Dict with "action_mask" and "phase".
        """
        super().reset(seed=seed)
        pass

    def step(self, action):
        """Apply action and advance environment state.

        Args:
            action: int (roll phase) or array [category, column] (score phase).

        Returns:
            obs, reward, terminated, truncated, info
        """
        pass

    def _get_obs(self) -> np.ndarray:
        """Construct flat observation vector from current state."""
        pass

    def _get_action_mask(self) -> np.ndarray:
        """Compute binary action mask for current phase and dice state."""
        pass

    def _is_terminal(self) -> bool:
        """Return True when all slots across all columns are filled."""
        pass
