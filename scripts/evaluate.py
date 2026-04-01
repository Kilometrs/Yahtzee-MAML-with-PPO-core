"""Evaluation entry point.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/5000.pt
    python scripts/evaluate.py --checkpoint checkpoints/5000.pt --config configs/server.yaml
    python scripts/evaluate.py --checkpoint checkpoints/5000.pt --n-episodes 200
    python scripts/evaluate.py --checkpoint checkpoints/5000.pt --task-name thesis-eval-run1
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained FOMAML checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="ClearML task name (default: eval-<checkpoint_stem>)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    task_name = args.task_name or f"eval-{Path(args.checkpoint).stem}"

    from logging_utils.clearml_logger import ClearMLLogger
    from tasks.reward_tasks import TASK_REGISTRY
    from training.evaluator import Evaluator

    logger = ClearMLLogger(
        project_name=config["clearml"]["project_name"],
        task_name=task_name,
        config=config,
    )

    try:
        evaluator = Evaluator(config)
        df_steps, df_episodes = evaluator.evaluate(
            checkpoint_path=args.checkpoint,
            n_episodes=args.n_episodes,
        )

        # Resolve meta_step from checkpoint for parquet naming
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        meta_step = ckpt.get("meta_step", 0)

        threshold = config["tasks"]["threshold_beater_score"]
        tasks = [
            TASK_REGISTRY["MaxScore"](),
            TASK_REGISTRY["ThresholdBeater"](threshold=threshold),
            TASK_REGISTRY["UpperBonus"](),
            TASK_REGISTRY["YahtzeeHunter"](),
            TASK_REGISTRY["Conservative"](),
        ]

        for task in tasks:
            task_steps = df_steps[df_steps["strategy"] == task.name]
            task_eps = df_episodes[df_episodes["strategy"] == task.name]
            steps_path, eps_path = evaluator.save_trajectories(
                task_steps, task_eps, task.name, meta_step
            )
            logger.log_artifact(
                name=f"trajectories_{task.name}_step{meta_step}",
                path=str(steps_path),
            )
            logger.log_artifact(
                name=f"episodes_{task.name}_step{meta_step}",
                path=str(eps_path),
            )
            print(f"Saved: {steps_path}")
            print(f"Saved: {eps_path}")

    except Exception as e:
        print(f"Evaluation failed: {e}", file=sys.stderr)
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
