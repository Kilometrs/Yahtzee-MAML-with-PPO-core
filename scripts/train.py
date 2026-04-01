"""Training entry point.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/server.yaml
    python scripts/train.py --config configs/default.yaml --resume checkpoints/500.pt
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Run FOMAML meta-training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = config["env"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from training import MetaTrainer

    trainer = MetaTrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
