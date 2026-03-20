"""Training entry point.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/server.yaml
"""

import argparse
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run FOMAML meta-training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from training import MetaTrainer
    trainer = MetaTrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
