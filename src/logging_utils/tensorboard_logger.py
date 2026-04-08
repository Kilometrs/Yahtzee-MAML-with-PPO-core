"""TensorBoard experiment logger — drop-in replacement for ClearMLLogger."""

import os
import time
from datetime import datetime

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


class TensorBoardLogger:
    """Local TensorBoard logger with the same interface as ClearMLLogger."""

    def __init__(self, project_name: str, task_name: str, config: dict):
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join("runs", project_name, f"{run_id}_{task_name}")
        os.makedirs(log_dir, exist_ok=True)
        self._writer = EventFileWriter(log_dir)
        self._log_dir = log_dir
        self._task_id = f"tb_{run_id}"
        print(f"TensorBoard logging to {log_dir}")
        print(f"  View with: tensorboard --logdir runs/")

    @property
    def task_id(self) -> str:
        return self._task_id

    def _add_scalar(self, tag: str, value: float, step: int):
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)])
        event = Event(wall_time=time.time(), step=step, summary=summary)
        self._writer.add_event(event)

    def log_scalar(self, title: str, series: str, value: float, step: int) -> None:
        self._add_scalar(f"{title}/{series}", value, step)

    def log_artifact(self, name: str, path: str) -> None:
        print(f"[TensorBoard] Artifact not uploaded (local only): {name} -> {path}")

    def log_plot(self, title: str, figure) -> None:
        path = os.path.join(self._log_dir, f"{title}.png")
        figure.savefig(path)
        print(f"[TensorBoard] Plot saved to {path}")

    def flush_scalars(self):
        self._writer.flush()

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()
        print("TensorBoard writer closed.")
