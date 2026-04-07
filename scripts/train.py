"""Training entry point for JAX Yahtzee FOMAML."""
import argparse
import os
import sys
import yaml
import numpy as np
import jax

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.meta_trainer import MetaTrainer


def _load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    _load_dotenv()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    np.random.seed(config["env"]["seed"])
    print(f"JAX devices: {jax.devices()}")
    print(f"Config: {args.config}")
    trainer = MetaTrainer(config)
    start_step = 0
    if args.resume:
        start_step = trainer.load_checkpoint(args.resume)
        print(f"Resumed from step {start_step}")
    try:
        trainer.train(start_step=start_step)
    finally:
        trainer.logger.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
