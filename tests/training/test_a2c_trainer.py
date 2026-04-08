"""Tests for standalone A2C trainer utilities and training loop."""
import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import optax

from src.training.a2c_trainer import build_lr_schedule, anneal_entropy, A2CTrainer

TINY_CONFIG = {
    "env": {"n_columns": 1, "seed": 42, "max_steps_per_episode": 50},
    "agent": {"hidden_dim": 32, "n_layers": 1, "use_layer_norm": False,
              "activation": "relu", "dropout_rate": 0.0, "head_hidden_dim": 0,
              "norm_position": "pre"},
    "a2c": {"gamma": 0.99, "gae_lambda": 0.0, "value_loss_coef": 0.5,
            "entropy_coef_roll": [0.1, 0.02], "entropy_coef_score": [0.03, 0.01],
            "entropy_hold": 0.075, "entropy_anneal": 0.9},
    "training": {"n_games": 20, "n_parallel_envs": 2, "lr": 0.001,
                 "lr_min_ratio": 0.05, "lr_warmup": 0.05, "lr_plateau": 0.70,
                 "lr_decay": 0.25, "checkpoint_every": 100, "eval_every": 0,
                 "n_eval_episodes": 0, "checkpoint_dir": "", "log_every": 0,
                 "device": "cpu"},
    "tasks": {"threshold_beater_score": 250},
    "clearml": {"project_name": "test", "task_name": "test"},
}


class TestLRSchedule:
    def test_warmup_starts_near_zero(self):
        schedule = build_lr_schedule(
            peak_lr=1e-4, min_ratio=0.05, warmup_frac=0.05,
            plateau_frac=0.70, total_steps=50000)
        assert schedule(0) < 1e-5

    def test_warmup_reaches_peak(self):
        schedule = build_lr_schedule(
            peak_lr=1e-4, min_ratio=0.05, warmup_frac=0.05,
            plateau_frac=0.70, total_steps=50000)
        warmup_end = int(50000 * 0.05)
        lr = schedule(warmup_end)
        assert abs(float(lr) - 1e-4) < 1e-6

    def test_plateau_is_constant(self):
        schedule = build_lr_schedule(
            peak_lr=1e-4, min_ratio=0.05, warmup_frac=0.05,
            plateau_frac=0.70, total_steps=50000)
        warmup_end = int(50000 * 0.05)
        plateau_end = warmup_end + int(50000 * 0.70)
        lr_start = float(schedule(warmup_end + 100))
        lr_end = float(schedule(plateau_end - 100))
        assert abs(lr_start - lr_end) < 1e-7

    def test_decay_reaches_min(self):
        schedule = build_lr_schedule(
            peak_lr=1e-4, min_ratio=0.05, warmup_frac=0.05,
            plateau_frac=0.70, total_steps=50000)
        lr_final = float(schedule(49999))
        expected_min = 1e-4 * 0.05
        assert abs(lr_final - expected_min) < 1e-6

    def test_usable_with_optax(self):
        schedule = build_lr_schedule(
            peak_lr=1e-4, min_ratio=0.05, warmup_frac=0.05,
            plateau_frac=0.70, total_steps=1000)
        optimizer = optax.adam(learning_rate=schedule)
        params = {"w": jnp.ones(3)}
        opt_state = optimizer.init(params)
        grads = {"w": jnp.ones(3) * 0.1}
        updates, new_state = optimizer.update(grads, opt_state, params)
        assert all(jnp.all(jnp.isfinite(v)) for v in jax.tree.leaves(updates))


class TestEntropyAnnealing:
    def test_hold_period_returns_max(self):
        val = anneal_entropy(step=0, total_steps=50000,
                             max_val=0.1, min_val=0.02,
                             hold_frac=0.075, anneal_frac=0.9)
        assert abs(float(val) - 0.1) < 1e-6

    def test_mid_anneal(self):
        total = 50000
        hold_end = int(total * 0.075)
        anneal_end = hold_end + int(total * 0.9)
        midpoint = (hold_end + anneal_end) // 2
        val = anneal_entropy(step=midpoint, total_steps=total,
                             max_val=0.1, min_val=0.02,
                             hold_frac=0.075, anneal_frac=0.9)
        expected = (0.1 + 0.02) / 2
        assert abs(float(val) - expected) < 0.01

    def test_floor_after_anneal(self):
        val = anneal_entropy(step=49999, total_steps=50000,
                             max_val=0.1, min_val=0.02,
                             hold_frac=0.075, anneal_frac=0.9)
        assert abs(float(val) - 0.02) < 1e-6

    def test_never_below_min(self):
        for step in [0, 10000, 25000, 49000, 49999]:
            val = anneal_entropy(step=step, total_steps=50000,
                                 max_val=0.1, min_val=0.02,
                                 hold_frac=0.075, anneal_frac=0.9)
            assert float(val) >= 0.02 - 1e-7


class TestA2CTrainer:
    def test_train_runs_without_error(self):
        os.environ["CLEARML_OFF"] = "1"
        trainer = A2CTrainer(TINY_CONFIG)
        losses = trainer.train()
        assert len(losses) == 10
        assert all(np.isfinite(l) for l in losses)

    def test_params_change_after_training(self):
        os.environ["CLEARML_OFF"] = "1"
        trainer = A2CTrainer(TINY_CONFIG)
        initial_leaves = [x.copy() for x in jax.tree.leaves(trainer.params)]
        trainer.train()
        final_leaves = jax.tree.leaves(trainer.params)
        any_changed = any(
            not jnp.array_equal(i, f)
            for i, f in zip(initial_leaves, final_leaves))
        assert any_changed


class TestA2CCheckpoint:
    def test_save_and_load_roundtrip(self, tmp_path):
        os.environ["CLEARML_OFF"] = "1"
        config = {**TINY_CONFIG}
        config["training"] = {**config["training"], "checkpoint_dir": str(tmp_path)}
        trainer = A2CTrainer(config)
        trainer.train()
        save_path = trainer._save_checkpoint(10)
        trainer2 = A2CTrainer(config)
        step = trainer2.load_checkpoint(save_path)
        assert step == 10
        for orig, loaded in zip(jax.tree.leaves(trainer.params),
                                jax.tree.leaves(trainer2.params)):
            assert jnp.allclose(orig, loaded)
