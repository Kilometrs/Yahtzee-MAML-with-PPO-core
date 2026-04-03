"""Five reward-shaped tasks for the FOMAML task distribution."""

from env.constants import UPPER_CATEGORIES, UPPER_BONUS_THRESHOLD, UPPER_BONUS_VALUE, YAHTZEE
from tasks.base_task import BaseTask


class MaxScore(BaseTask):
    """Maximize the raw final score across all columns."""

    @property
    def name(self) -> str:
        return "MaxScore"

    def reward(self, state, action, next_state, done, info) -> float:
        """Return normalised score delta per step (includes Yahtzee bonus if triggered).

        Divided by 50.0 (max single-slot score) to keep reward scale comparable
        to other tasks which return values in the 0-1 range.
        """
        if state["phase"] != "score":
            return 0.0
        return float(next_state["total_score"] - state["total_score"]) / 50.0


class ThresholdBeater(BaseTask):
    """Maximize probability of exceeding a score threshold."""

    def __init__(self, threshold: int = 250):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "ThresholdBeater"

    def reward(self, state, action, next_state, done, info) -> float:
        """Return 1.0 on episode end if final_score >= threshold, else 0.0."""
        if done:
            return 1.0 if next_state["total_score"] >= self.threshold else 0.0
        return 0.0


class UpperBonus(BaseTask):
    """Maximize upper section bonus completion (63-point threshold per column)."""

    @property
    def name(self) -> str:
        return "UpperBonus"

    def reward(self, state, action, next_state, done, info) -> float:
        """Reward upper section placements; bonus when 63-pt threshold crossed."""
        if state["phase"] != "score":
            return 0.0

        cat, col = int(action[0]), int(action[1])
        if cat not in UPPER_CATEGORIES:
            return 0.0

        score_gained = next_state["scores"][col][cat]
        reward = score_gained / 3.0

        # Check if the upper bonus was newly triggered for this column
        prev_upper = sum(state["scores"][col][c] for c in UPPER_CATEGORIES)
        new_upper = sum(next_state["scores"][col][c] for c in UPPER_CATEGORIES)
        if prev_upper < UPPER_BONUS_THRESHOLD <= new_upper:
            reward += float(UPPER_BONUS_VALUE)

        return reward


class YahtzeeHunter(BaseTask):
    """Maximize Yahtzee frequency — aggressive high-variance play."""

    @property
    def name(self) -> str:
        return "YahtzeeHunter"

    def reward(self, state, action, next_state, done, info) -> float:
        """Large reward per Yahtzee scored (50 pts) and per bonus Yahtzee (+100)."""
        if state["phase"] != "score":
            return 0.0

        cat, col = int(action[0]), int(action[1])
        reward = 0.0

        if cat == YAHTZEE:
            score_gained = next_state["scores"][col][cat]
            if score_gained > 0:  # score_category returns 50 or 0 for YAHTZEE
                reward += 50.0

        # env increments yahtzee_bonus by YAHTZEE_BONUS_VALUE (100) per bonus triggered
        yahtzee_bonus_delta = next_state["yahtzee_bonus"] - state["yahtzee_bonus"]
        reward += float(yahtzee_bonus_delta)

        return reward


class Conservative(BaseTask):
    """Minimize zeros and cross-outs — consistent, safe play."""

    @property
    def name(self) -> str:
        return "Conservative"

    def reward(self, state, action, next_state, done, info) -> float:
        """Penalize cross-outs and zero scores; reward consistent placements."""
        if state["phase"] != "score":
            return 0.0

        cat, col = int(action[0]), int(action[1])
        score_gained = next_state["scores"][col][cat]
        return 1.0 if score_gained > 0 else -1.0


TASK_REGISTRY: dict[str, type[BaseTask]] = {
    "MaxScore": MaxScore,
    "ThresholdBeater": ThresholdBeater,
    "UpperBonus": UpperBonus,
    "YahtzeeHunter": YahtzeeHunter,
    "Conservative": Conservative,
}
