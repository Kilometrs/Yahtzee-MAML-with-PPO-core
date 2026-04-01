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
from agents.actor_critic import ActorCritic
from agents.rollout_buffer import RolloutBuffer


def clone_params(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a cloned copy of all model parameters with requires_grad=True.

    The cloned tensors require grad so that the outer loop can compute
    meta-gradients through the inner update.

    Args:
        model: PyTorch module.

    Returns:
        Dict mapping parameter name -> cloned tensor (requires_grad=True).
    """
    return {k: v.clone().requires_grad_(True) for k, v in model.named_parameters()}


def collect_episode(
    model: ActorCritic,
    env,
    device: torch.device,
    params: dict[str, torch.Tensor] | None = None,
) -> RolloutBuffer:
    """Run one complete Yahtzee episode and return the filled RolloutBuffer.

    Args:
        model: ActorCritic policy.
        env: YahtzeeEnv instance (will be reset at start).
        device: Torch device for tensor ops.
        params: If provided, temporarily load these weights into model before
                the episode and restore originals afterward (param-swap pattern
                for query rollout collection with adapted fast_params).

    Returns:
        Filled RolloutBuffer with one episode of transitions.
    """
    original_state = None
    if params is not None:
        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(params)

    try:
        buf = RolloutBuffer()
        obs, info = env.reset()

        while True:
            phase = info["phase"]
            mask = info["action_mask"]

            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask, dtype=torch.float32, device=device)

            with torch.no_grad():
                action, log_prob = model.get_action(obs_t, phase, mask_t)
                value = model.get_value(obs_t)

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            buf.add(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                log_prob=log_prob.item(),
                value=value.squeeze().item(),
                phase=phase,
                action_mask=mask,
            )

            if done:
                break

            obs = next_obs
            info = next_info

        return buf

    finally:
        if original_state is not None:
            model.load_state_dict(original_state)


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
    """Perform one FOMAML inner loop gradient step. Phase 2."""
    raise NotImplementedError("Phase 2")
