"""Evaluation entry point.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/step5000.pt --config configs/default.yaml
"""

import argparse
import yaml


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained FOMAML checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n-episodes", type=int, default=100)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from training import Evaluator
    evaluator = Evaluator(config)
    evaluator.evaluate(args.checkpoint, n_episodes=args.n_episodes)


if __name__ == "__main__":
    main()
