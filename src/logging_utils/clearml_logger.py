"""ClearML experiment logger wrapper."""


class ClearMLLogger:
    """Thin wrapper around ClearML Task for structured metric/artifact logging.

    Initialised once per training run. Handles:
    - Scalar metrics (loss curves, reward curves, rates)
    - File artifacts (checkpoints, parquet trajectory files)
    - Matplotlib figures (score distributions, entropy plots)
    """

    def __init__(self, project_name: str, task_name: str, config: dict):
        """
        Args:
            project_name: ClearML project name (from config or env var).
            task_name: ClearML task name for this run.
            config: Full config dict — logged as a config artifact.
        """
        pass

    def log_scalar(self, title: str, series: str, value: float, step: int) -> None:
        """Log a scalar value to ClearML.

        Args:
            title: Plot title (e.g. "Loss").
            series: Series name within plot (e.g. "meta_loss").
            value: Scalar value.
            step: Training step.
        """
        pass

    def log_artifact(self, name: str, path: str) -> None:
        """Upload a file as a ClearML artifact.

        Args:
            name: Artifact name (e.g. "trajectories_MaxScore_step5000").
            path: Local file path to upload.
        """
        pass

    def log_plot(self, title: str, figure) -> None:
        """Upload a matplotlib figure as a ClearML debug image.

        Args:
            title: Plot title shown in ClearML UI.
            figure: matplotlib Figure object.
        """
        pass

    def close(self) -> None:
        """Finalise and close the ClearML task."""
        pass
