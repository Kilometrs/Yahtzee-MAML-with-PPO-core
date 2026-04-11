"""ClearML experiment logger wrapper."""

from collections import defaultdict
from datetime import datetime

from clearml import Task


class ClearMLLogger:
    """Thin wrapper around ClearML Task for structured metric/artifact logging.

    Initialised once per training run. Handles:
    - Scalar metrics (loss curves, reward curves, rates)
    - File artifacts (checkpoints, parquet trajectory files)
    - Matplotlib figures (score distributions, entropy plots)

    Scalars are aggregated locally and flushed every ``log_every`` steps
    to avoid overwhelming ClearML's upload thread.  Each flush reports
    the **mean** value over the window, so no data is silently dropped.
    """

    def __init__(self, project_name: str, task_name: str, config: dict):
        """
        Args:
            project_name: ClearML project name (from config or env var).
            task_name: ClearML task name for this run.
            config: Full config dict — logged as a config artifact.
        """
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        full_task_name = f"{run_id}_{task_name}"
        task = Task.init(
            project_name=project_name,
            task_name=full_task_name,
            reuse_last_task_id=False,
        )
        task.connect(config)
        self._task = task
        self._logger = task.get_logger()
        self._log_every = config["training"].get("log_every", 100)
        # Buffers: (title, series) -> list of (step, value)
        self._scalar_buf = defaultdict(list)

    @property
    def task_id(self) -> str:
        return self._task.id

    def log_scalar(self, title: str, series: str, value: float, step: int) -> None:
        """Buffer a scalar value and flush every 10 entries per key.

        Args:
            title: Plot title (e.g. "Loss").
            series: Series name within plot (e.g. "meta_loss").
            value: Scalar value.
            step: Training step.
        """
        key = (title, series)
        self._scalar_buf[key].append((step, value))
        if len(self._scalar_buf[key]) >= 10:
            self._flush_scalar(key)

    def _flush_scalar(self, key):
        buf = self._scalar_buf.pop(key, [])
        if not buf:
            return
        title, series = key
        for step, value in buf:
            self._logger.report_scalar(
                title=title, series=series, value=value, iteration=step)

    def flush_scalars(self):
        """Flush all buffered scalars (call at end of training)."""
        for key in list(self._scalar_buf):
            self._flush_scalar(key)

    def log_artifact(self, name: str, path: str) -> None:
        """Upload a file as a ClearML artifact.

        Args:
            name: Artifact name (e.g. "trajectories_MaxScore_step5000").
            path: Local file path to upload.
        """
        self._task.upload_artifact(name=name, artifact_object=path)

    def log_plot(self, title: str, figure) -> None:
        """Upload a matplotlib figure as a ClearML debug image.

        Args:
            title: Plot title shown in ClearML UI.
            figure: matplotlib Figure object.
        """
        self._logger.report_matplotlib_figure(title=title, series=title, figure=figure, iteration=0)

    def close(self) -> None:
        """Flush all pending data and close the ClearML task."""
        print("Flushing metrics...")
        self.flush_scalars()
        self._logger.flush()
        print("Waiting for artifact uploads...")
        self._task.flush(wait_for_uploads=True)
        print(f"Closing ClearML task {self._task.id}")
        self._task.close()
        print("ClearML task closed.")
