"""Meta-training orchestration — outer FOMAML loop with checkpointing."""

from pathlib import Path


class MetaTrainer:
    """Orchestrates the FOMAML outer training loop.

    Responsibilities:
    - Build meta-policy, env factory, task distribution from config
    - Run outer loop for n_meta_steps
    - Save checkpoints every checkpoint_every steps
    - Log scalars and artifacts via ClearMLLogger
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Full config dict loaded from YAML (see configs/default.yaml).
        """
        pass

    def train(self) -> None:
        """Run full meta-training loop."""
        pass

    def save_checkpoint(self, step: int) -> Path:
        """Save checkpoint to config checkpoint_dir / step.pt.

        Checkpoint contains: meta_params, outer_optimizer_state,
        meta_step, config, task_names.

        Returns:
            Path to saved checkpoint file.
        """
        pass

    def load_checkpoint(self, path: str | Path) -> int:
        """Load checkpoint and restore training state.

        Returns:
            meta_step: The step count at time of checkpoint.
        """
        pass
