"""Abstract base class for MAML reward-shaping tasks."""

from abc import ABC, abstractmethod


class BaseTask(ABC):
    """A single task in the FOMAML task distribution.

    Each task defines a reward function that wraps the Yahtzee environment.
    The five reward tasks encode different strategic objectives:
        MaxScore:       maximize raw final score
        ThresholdBeater: maximize P(score >= threshold)
        UpperBonus:     maximize upper section bonus completion
        YahtzeeHunter:  maximize Yahtzee frequency (aggressive play)
        Conservative:   minimize zeros/scratches (consistent play)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable task name, used for logging and file naming."""
        pass

    @abstractmethod
    def reward(
        self,
        state: dict,
        action,
        next_state: dict,
        done: bool,
        info: dict,
    ) -> float:
        """Compute task-specific reward for one env transition.

        Args:
            state: Dict of env state before action (dice, scores, phase, etc.).
            action: Action taken.
            next_state: Dict of env state after action.
            done: Whether the episode ended.
            info: Env info dict (includes action_mask, cross_out flag, etc.).

        Returns:
            Scalar reward float.
        """
        pass
