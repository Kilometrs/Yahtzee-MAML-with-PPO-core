"""Training entry point.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/server.yaml
    python scripts/train.py --config configs/default.yaml --resume checkpoints/500.pt
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def _load_dotenv(path: Path) -> None:
    """Load key=value pairs from a .env file into os.environ (no overwrite)."""
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    _load_dotenv(Path(__file__).resolve().parent.parent / ".env")

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
