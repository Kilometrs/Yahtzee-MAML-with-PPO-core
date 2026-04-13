"""Entropy annealing for A2C (Pape 2025, Table 6)."""
import jax.numpy as jnp


def anneal_entropy(step, total_steps, max_val, min_val, hold_frac, anneal_frac):
    """Linear entropy annealing: hold -> decay -> floor.

    Pure JAX — safe inside JIT.

    Args:
        step: Current training step.
        total_steps: Total number of gradient updates.
        max_val: Starting (maximum) entropy coefficient.
        min_val: Floor (minimum) entropy coefficient.
        hold_frac: Fraction of training to hold at max_val.
        anneal_frac: Fraction of training for linear decay.
    """
    hold_end = total_steps * hold_frac
    anneal_end = hold_end + total_steps * anneal_frac

    progress = (step - hold_end) / jnp.maximum(anneal_end - hold_end, 1.0)
    progress = jnp.clip(progress, 0.0, 1.0)
    annealed = max_val + (min_val - max_val) * progress

    return jnp.where(step < hold_end, max_val, annealed)
