"""Shared utilities for RL loss computation. All functions are pure and jit-compatible."""
import jax
import jax.numpy as jnp


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    next_values = jnp.concatenate([values[1:], jnp.zeros(1)])

    def scan_fn(last_adv, t):
        reward, value, next_value, done = t
        delta = reward + gamma * next_value * (1.0 - done) - value
        adv = delta + gamma * lam * (1.0 - done) * last_adv
        return adv, adv

    _, advantages = jax.lax.scan(
        scan_fn, 0.0,
        (rewards[::-1], values[::-1], next_values[::-1], dones[::-1]),
    )
    advantages = advantages[::-1]
    returns = advantages + values
    return advantages, returns
