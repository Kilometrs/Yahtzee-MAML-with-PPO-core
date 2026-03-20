"""Five reward-shaped tasks for the FOMAML task distribution."""

from tasks.base_task import BaseTask


class MaxScore(BaseTask):
    """Maximize the raw final score across all columns."""

    @property
    def name(self) -> str:
        return "MaxScore"

    def reward(self, state, action, next_state, done, info) -> float:
        """Return score delta per step; large bonus on episode end."""
        pass


class ThresholdBeater(BaseTask):
    """Maximize probability of exceeding a score threshold."""

    def __init__(self, threshold: int = 250):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "ThresholdBeater"

    def reward(self, state, action, next_state, done, info) -> float:
        """Return 1.0 on episode end if final_score >= threshold, else 0.0."""
        pass


class UpperBonus(BaseTask):
    """Maximize upper section bonus completion (63-point threshold per column)."""

    @property
    def name(self) -> str:
        return "UpperBonus"

    def reward(self, state, action, next_state, done, info) -> float:
        """Reward upper section placements; bonus when threshold crossed."""
        pass


class YahtzeeHunter(BaseTask):
    """Maximize Yahtzee frequency — aggressive high-variance play."""

    @property
    def name(self) -> str:
        return "YahtzeeHunter"

    def reward(self, state, action, next_state, done, info) -> float:
        """Large reward per Yahtzee scored (50 pts) and per bonus Yahtzee (+100)."""
        pass


class Conservative(BaseTask):
    """Minimize zeros and cross-outs — consistent, safe play."""

    @property
    def name(self) -> str:
        return "Conservative"

    def reward(self, state, action, next_state, done, info) -> float:
        """Penalize cross-outs and zero scores; reward consistent placements."""
        pass


TASK_REGISTRY: dict[str, type[BaseTask]] = {
    "MaxScore": MaxScore,
    "ThresholdBeater": ThresholdBeater,
    "UpperBonus": UpperBonus,
    "YahtzeeHunter": YahtzeeHunter,
    "Conservative": Conservative,
}
