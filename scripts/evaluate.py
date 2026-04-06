"""Evaluation entry point.

Usage:
    python scripts/evaluate.py --checkpoint /mnt/checkpoints/step_5000 --config configs/default.yaml
    python scripts/evaluate.py --checkpoint /mnt/checkpoints/step_5000 --n-episodes 200
    python scripts/evaluate.py --checkpoint /mnt/checkpoints/step_5000 --task-name thesis-eval-run1
"""

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_params(checkpoint_dir, model, config):
    """Load params from a checkpoint directory, return (params, meta_step)."""
    import jax
    import jax.numpy as jnp

    meta_data = np.load(os.path.join(checkpoint_dir, "meta.npz"), allow_pickle=True)
    meta_step = int(meta_data["meta_step"])

    # Init dummy params to get pytree structure
    rng = jax.random.PRNGKey(0)
    obs_dim = 7 + 26 * config["env"]["n_columns"]
    ref_params = model.init(rng, jnp.zeros(obs_dim), jnp.int32(0))

    params_data = dict(np.load(os.path.join(checkpoint_dir, "params.npz"), allow_pickle=True))
    flat_params = [
        jnp.array(params_data["/".join(str(p) for p in k)])
        for k, _ in jax.tree_util.tree_leaves_with_path(ref_params)
    ]
    params = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(ref_params), flat_params)
    return params, meta_step


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained FOMAML checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory (e.g. /mnt/checkpoints/step_5000)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--task-name", type=str, default=None,
                        help="ClearML task name (default: eval-<checkpoint_stem>)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from src.agents.actor_critic import ActorCritic
    from src.logging_utils.clearml_logger import ClearMLLogger
    from src.tasks.reward_tasks import TASK_NAMES
    from src.training.evaluator import Evaluator

    checkpoint_stem = os.path.basename(args.checkpoint.rstrip("/"))
    task_name = args.task_name or f"eval-{checkpoint_stem}"

    model = ActorCritic(
        hidden_dim=config["agent"]["hidden_dim"],
        n_layers=config["agent"]["n_layers"],
        n_columns=config["env"]["n_columns"])

    params, meta_step = load_params(args.checkpoint, model, config)
    print(f"Loaded checkpoint: meta_step={meta_step}")

    logger = ClearMLLogger(
        project_name=config["clearml"]["project_name"],
        task_name=task_name,
        config=config,
    )

    try:
        evaluator = Evaluator(config)
        df_steps, df_episodes = evaluator.evaluate(
            params, meta_step, n_episodes=args.n_episodes)

        for task_name in TASK_NAMES:
            task_steps = df_steps[df_steps["strategy"] == task_name]
            task_eps = df_episodes[df_episodes["strategy"] == task_name]
            steps_path, eps_path = evaluator.save_trajectories(
                task_steps, task_eps, task_name, meta_step)
            logger.log_artifact(
                name=f"trajectories_{task_name}_step{meta_step}",
                path=str(steps_path))
            logger.log_artifact(
                name=f"episodes_{task_name}_step{meta_step}",
                path=str(eps_path))
            mean_score = task_eps["final_score"].mean()
            print(f"{task_name}: mean_score={mean_score:.1f}  saved: {steps_path}")

    except Exception as e:
        print(f"Evaluation failed: {e}", file=sys.stderr)
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
