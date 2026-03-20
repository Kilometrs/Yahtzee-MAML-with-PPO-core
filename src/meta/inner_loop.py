"""FOMAML inner loop — stateless parameter cloning and gradient update.

Uses torch.func.functional_call for differentiable forward passes on
cloned parameters without modifying the original model in-place.

FOMAML inner loop pattern:
    1. fast_params = clone_params(model)
    2. For each inner step:
       a. loss = compute_ppo_loss using functional_call(model, fast_params, obs)
       b. grads = torch.autograd.grad(loss, fast_params.values())
       c. fast_params = {k: v - inner_lr * g for (k, v), g in zip(...)}
    3. Return fast_params (used by outer loop for meta-gradient)
"""

import torch
from torch.func import functional_call
import torch.nn as nn


def clone_params(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a detached copy of all model parameters.

    The cloned tensors require grad so that the outer loop can compute
    meta-gradients through the inner update.

    Args:
        model: PyTorch module.

    Returns:
        Dict mapping parameter name -> cloned tensor (requires_grad=True).
    """
    pass


def inner_update(
    model: nn.Module,
    fast_params: dict[str, torch.Tensor],
    obs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    log_probs_old: torch.Tensor,
    phases: list[str],
    inner_lr: float,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Perform one FOMAML inner loop gradient step.

    Computes PPO clipped surrogate loss using functional_call on fast_params,
    then returns updated fast_params (first-order — no second-order gradients).

    Args:
        model: The meta-policy model (structure reference only, not updated).
        fast_params: Current task-adapted parameter dict.
        obs: Observation tensor (T, obs_dim).
        actions: Action tensor (T,).
        advantages: GAE advantage estimates (T,).
        log_probs_old: Log probabilities from rollout collection (T,).
        phases: List of phase strings ("roll" or "score") per step.
        inner_lr: Inner loop learning rate.
        clip_epsilon: PPO clip coefficient.
        entropy_coef: Entropy regularisation coefficient.

    Returns:
        Updated fast_params dict.
    """
    pass
