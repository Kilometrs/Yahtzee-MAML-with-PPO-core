"""Tests for entropy annealing."""
import jax.numpy as jnp
import pytest
from src.agents.entropy import anneal_entropy


class TestAnnealEntropy:
    def test_hold_period(self):
        val = anneal_entropy(step=0, total_steps=1000,
                             max_val=0.1, min_val=0.02,
                             hold_frac=0.1, anneal_frac=0.9)
        assert float(val) == pytest.approx(0.1, abs=1e-6)

    def test_end_of_hold(self):
        val = anneal_entropy(step=99, total_steps=1000,
                             max_val=0.1, min_val=0.02,
                             hold_frac=0.1, anneal_frac=0.9)
        assert float(val) == pytest.approx(0.1, abs=1e-6)

    def test_mid_anneal(self):
        val = anneal_entropy(step=550, total_steps=1000,
                             max_val=0.1, min_val=0.02,
                             hold_frac=0.1, anneal_frac=0.9)
        assert 0.02 < float(val) < 0.1

    def test_floor(self):
        val = anneal_entropy(step=999, total_steps=1000,
                             max_val=0.1, min_val=0.02,
                             hold_frac=0.1, anneal_frac=0.9)
        assert float(val) == pytest.approx(0.02, abs=1e-3)

    def test_jax_scalar_input(self):
        val = anneal_entropy(step=jnp.int32(50), total_steps=1000,
                             max_val=0.1, min_val=0.02,
                             hold_frac=0.1, anneal_frac=0.9)
        assert jnp.isfinite(val)
