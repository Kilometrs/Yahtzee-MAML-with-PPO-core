"""Evaluation — fine-tune meta-policy per task, collect trajectories, save parquet."""

from pathlib import Path
import pandas as pd


class Evaluator:
    """Evaluates a trained meta-policy checkpoint.

    For each task:
    1. Load meta-policy from checkpoint
    2. Fine-tune for M inner loop steps on the task
    3. Run N evaluation episodes, collecting full per-step trajectory data
    4. Save trajectory and episode summary as parquet files
    5. Upload to ClearML as artifacts

    Trajectory parquet schema (per-step):
        episode (int), round (int), rerolls_left (int), dice (list[int]),
        phase (str), action_category (int), action_column (int),
        score_gained (int), value_estimate (float), entropy (float),
        action_probs (list[float]), cross_out (bool), yahtzee_bonus_event (bool),
        cumulative_score (int), strategy (str), meta_step (int)

    Episode summary parquet schema:
        episode (int), final_score (int), n_cross_outs (int),
        n_yahtzees (int), yahtzee_bonus_total (int),
        upper_bonus_earned (bool), strategy (str), meta_step (int)
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Full config dict loaded from YAML.
        """
        pass

    def evaluate(
        self,
        checkpoint_path: str | Path,
        n_episodes: int = 100,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run evaluation for all tasks.

        Returns:
            (df_steps, df_episodes): Per-step trajectory and per-episode summary DataFrames.
        """
        pass

    def save_trajectories(
        self,
        df_steps: pd.DataFrame,
        df_episodes: pd.DataFrame,
        strategy: str,
        meta_step: int,
    ) -> tuple[Path, Path]:
        """Save DataFrames to parquet under data/trajectories/ and data/episodes/.

        Returns:
            (steps_path, episodes_path): Paths to saved parquet files.
        """
        pass
