"""Multi-column Yahtzee Gymnasium environment."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.constants import (
    N_DICE, N_CATEGORIES, MAX_REROLLS,
    YAHTZEE, YAHTZEE_BONUS_VALUE,
    UPPER_CATEGORIES, UPPER_BONUS_THRESHOLD, UPPER_BONUS_VALUE,
)
from env.scoring import score_category, compute_valid_actions


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

    _phase is the sole source of truth for current phase. It is surfaced
    in info["phase"] every step. remaining_rerolls in the obs is
    informational only and is never zeroed for phase signalling.

    Action masks are returned in info["action_mask"].
    The mask shape changes with phase: roll → (32,), score → (13 * n_columns,).
    Always read info["phase"] alongside info["action_mask"] to interpret the mask.
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

        # Gymnasium protocol requires action_space to be set.
        # The effective action space is phase-dependent; use info["action_mask"] for valid actions.
        self.action_space = self.roll_action_space

        # Internal state (populated on reset)
        self._dice = None
        self._remaining_rerolls = None
        self._filled_mask = None   # shape: (n_columns, N_CATEGORIES)
        self._scores = None        # shape: (n_columns, N_CATEGORIES)
        self._yahtzee_bonus = None
        self._phase = None         # "roll" or "score"

    def reset(self, seed=None, options=None):
        """Reset environment to a new game."""
        super().reset(seed=seed)
        self._dice = self.np_random.integers(1, 7, size=N_DICE)
        self._remaining_rerolls = MAX_REROLLS
        self._filled_mask = np.zeros((self.n_columns, N_CATEGORIES), dtype=bool)
        self._scores = np.zeros((self.n_columns, N_CATEGORIES), dtype=np.int32)
        self._yahtzee_bonus = 0
        self._phase = "roll"
        info = {"action_mask": self._get_action_mask(), "phase": self._phase}
        return self._get_obs(), info

    def step(self, action):
        """Apply action and advance environment state."""
        if self._phase == "roll":
            return self._step_roll(action)
        return self._step_score(action)

    def _step_roll(self, action):
        action = int(action)
        if action == 31:
            # Keep all dice: voluntarily transition to score phase.
            # _remaining_rerolls is NOT changed — _phase is the source of truth.
            self._phase = "score"
        else:
            # Reroll dice whose bit is 0 in the keep-mask
            for i in range(N_DICE):
                if not (action >> i) & 1:
                    self._dice[i] = self.np_random.integers(1, 7)
            self._remaining_rerolls -= 1
            if self._remaining_rerolls == 0:
                self._phase = "score"
        info = {
            "action_mask": self._get_action_mask(),
            "phase": self._phase,
            "cross_out": False,
        }
        return self._get_obs(), 0.0, False, False, info

    def _step_score(self, action):
        category, column = int(action[0]), int(action[1])

        # Capture state before any mutation; next_state_dict captured after.
        state_dict = self._make_state_dict()

        score_gained = score_category(self._dice.tolist(), category)

        # Yahtzee bonus: five-of-a-kind AND all Yahtzee slots already filled at 50
        if len(set(self._dice.tolist())) == 1:
            all_yahtzee_filled = all(
                self._filled_mask[col][YAHTZEE] and self._scores[col][YAHTZEE] == 50
                for col in range(self.n_columns)
            )
            if all_yahtzee_filled:
                self._yahtzee_bonus += YAHTZEE_BONUS_VALUE

        self._scores[column][category] = score_gained
        self._filled_mask[column][category] = True

        terminated = self._is_terminal()

        if not terminated:
            self._dice = self.np_random.integers(1, 7, size=N_DICE)
            self._remaining_rerolls = MAX_REROLLS
            self._phase = "roll"

        next_state_dict = self._make_state_dict()
        cross_out = score_gained == 0

        info = {
            "action_mask": self._get_action_mask(),
            "phase": self._phase,
            "cross_out": cross_out,
        }

        if self.reward_fn is not None:
            reward = float(self.reward_fn(state_dict, action, next_state_dict, terminated, info))
        else:
            reward = float(next_state_dict["total_score"] - state_dict["total_score"])

        return self._get_obs(), reward, terminated, False, info

    def _make_state_dict(self) -> dict:
        """Construct the state dict passed to reward_fn."""
        return {
            "dice": self._dice.tolist(),
            "remaining_rerolls": int(self._remaining_rerolls),
            "filled_mask": [row.tolist() for row in self._filled_mask],
            "scores": [row.tolist() for row in self._scores],
            "yahtzee_bonus": int(self._yahtzee_bonus),
            "phase": self._phase,
            "total_score": self._compute_total_score(),
            "upper_score": self._compute_upper_score(),
        }

    def _compute_upper_score(self) -> int:
        return int(sum(
            self._scores[col][cat]
            for col in range(self.n_columns)
            for cat in UPPER_CATEGORIES
        ))

    def _compute_upper_bonuses(self) -> int:
        total = 0
        for col in range(self.n_columns):
            col_upper = sum(self._scores[col][cat] for cat in UPPER_CATEGORIES)
            if col_upper >= UPPER_BONUS_THRESHOLD:
                total += UPPER_BONUS_VALUE
        return total

    def _compute_total_score(self) -> int:
        return (
            int(np.sum(self._scores))
            + self._compute_upper_bonuses()
            + int(self._yahtzee_bonus)
        )

    def _get_obs(self) -> np.ndarray:
        """Construct flat observation vector from current state."""
        return np.concatenate([
            self._dice.astype(np.float32),
            np.array([self._remaining_rerolls], dtype=np.float32),
            self._filled_mask.flatten().astype(np.float32),
            self._scores.flatten().astype(np.float32),
            np.array([self._yahtzee_bonus], dtype=np.float32),
        ])

    def _get_action_mask(self) -> np.ndarray:
        """Compute binary action mask for current phase and dice state."""
        if self._phase == "roll":
            return np.ones(32, dtype=np.float32)
        # Score phase: mark valid (cat, col) slots
        mask = np.zeros(N_CATEGORIES * self.n_columns, dtype=np.float32)
        valid = compute_valid_actions(
            self._dice.tolist(),
            [row.tolist() for row in self._filled_mask],
        )
        for cat, col in valid:
            mask[cat * self.n_columns + col] = 1.0
        return mask

    def _is_terminal(self) -> bool:
        """Return True when all slots across all columns are filled."""
        return bool(np.all(self._filled_mask))
