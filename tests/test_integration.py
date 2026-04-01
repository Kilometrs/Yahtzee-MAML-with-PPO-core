"""Integration test: 10 meta-steps complete without error and produce a checkpoint."""

import math
import random
from pathlib import Path

import torch
import yaml

from agents.actor_critic import ActorCritic
from env.yahtzee_env import YahtzeeEnv
from meta.maml import FOMAML
from tasks.reward_tasks import TASK_REGISTRY
from training.evaluator import Evaluator


def test_10_meta_steps_and_checkpoint(tmp_path):
    """Run 10 FOMAML meta-steps; save a checkpoint; load it in Evaluator."""
    with open("configs/test.yaml") as f:
        config = yaml.safe_load(f)

    config["training"]["checkpoint_dir"] = str(tmp_path)

    n_columns = config["env"]["n_columns"]
    obs_dim = 7 + 26 * n_columns
    device = torch.device("cpu")

    model = ActorCritic(
        obs_dim=obs_dim,
        n_columns=n_columns,
        hidden_dim=config["agent"]["hidden_dim"],
        n_layers=config["agent"]["n_layers"],
    ).to(device)

    fomaml = FOMAML(
        model=model,
        inner_lr=config["meta"]["inner_lr"],
        outer_lr=config["meta"]["outer_lr"],
        n_inner_steps=config["meta"]["n_inner_steps"],
    )

    threshold = config["tasks"]["threshold_beater_score"]
    tasks = [
        TASK_REGISTRY["MaxScore"](),
        TASK_REGISTRY["ThresholdBeater"](threshold=threshold),
        TASK_REGISTRY["UpperBonus"](),
        TASK_REGISTRY["YahtzeeHunter"](),
        TASK_REGISTRY["Conservative"](),
    ]

    ppo_cfg = config["ppo"]
    n_tasks_per_batch = config["meta"]["n_tasks_per_batch"]

    losses = []
    for step in range(10):
        task_batch = random.sample(tasks, min(n_tasks_per_batch, len(tasks)))
        loss = fomaml.meta_update(
            tasks=task_batch,
            env_fn=lambda: YahtzeeEnv(n_columns=n_columns),
            ppo_cfg=ppo_cfg,
            device=device,
        )
        assert math.isfinite(loss), f"meta_loss not finite at step {step}: {loss}"
        losses.append(loss)

    assert len(losses) == 10

    # Save checkpoint
    ckpt_path = tmp_path / "10.pt"
    torch.save(
        {
            "meta_params": model.state_dict(),
            "outer_optimizer_state": fomaml.meta_optimizer.state_dict(),
            "meta_step": 10,
            "config": config,
            "task_names": [t.name for t in tasks],
        },
        ckpt_path,
    )
    assert ckpt_path.exists()

    # Evaluator can load the checkpoint and run 2 eval episodes per task
    evaluator = Evaluator(config)
    df_steps, df_episodes = evaluator.evaluate(ckpt_path, n_episodes=2)
    assert len(df_episodes) == len(tasks) * 2
    assert "strategy" in df_episodes.columns
    assert "meta_step" in df_episodes.columns
