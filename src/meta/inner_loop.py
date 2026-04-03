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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
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


def _ppo_loss_fn(
    model: nn.Module,
    fast_params: dict[str, torch.Tensor],
    obs: torch.Tensor,
    actions,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    log_probs_old: torch.Tensor,
    phases,
    action_masks,
    clip_epsilon: float,
    entropy_coef: float,
    value_loss_coef: float,
) -> torch.Tensor:
    """Compute PPO-style loss using functional_call with fast_params.

    Processes both roll and score phases separately, sums losses, normalises
    by total step count. Used by inner_update and FOMAML.meta_update.
    Does not modify model weights — all forward passes are via functional_call.

    Returns:
        Scalar loss tensor with grad_fn attached to fast_params.
    """
    loss_terms = []
    total_count = 0

    for phase_str in ("roll", "score"):
        pmask_np = phases == phase_str
        if not pmask_np.any():
            continue
        pmask = torch.from_numpy(pmask_np)

        p_obs = obs[pmask]
        p_adv = advantages[pmask]
        p_returns = returns[pmask]
        p_old_lp = log_probs_old[pmask]
        p_actions = actions[pmask_np]
        p_masks_np = action_masks[pmask_np]

        # Forward pass with injected fast_params — does not modify model weights
        logits, values_pred = functional_call(model, fast_params, (p_obs, phase_str))

        mask_tensor = torch.tensor(
            np.array(list(p_masks_np), dtype=np.float32),
            dtype=torch.float32,
        ).to(obs.device)
        logits = logits.masked_fill(mask_tensor == 0, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)

        if phase_str == "score":
            acts_np = np.array(list(p_actions))   # shape (N, 2)
            flat = acts_np[:, 0] * model.n_columns + acts_np[:, 1]
            act_tensor = torch.tensor(flat, dtype=torch.long).to(obs.device)
        else:
            act_tensor = torch.tensor(list(p_actions), dtype=torch.long).to(obs.device)

        new_lp = dist.log_prob(act_tensor)
        entropy = dist.entropy()

        ratio = torch.exp(new_lp - p_old_lp)
        clip_adv = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * p_adv
        policy_loss = -torch.min(ratio * p_adv, clip_adv).sum()
        value_loss = 0.5 * F.mse_loss(
            values_pred.squeeze(-1), p_returns, reduction="sum"
        )
        entropy_loss = -entropy_coef * entropy.sum()

        loss_terms.append(policy_loss + value_loss_coef * value_loss + entropy_loss)
        total_count += int(pmask_np.sum())

    if total_count == 0:
        return torch.tensor(0.0)

    return torch.stack(loss_terms).sum() / total_count


def inner_update(
    model: nn.Module,
    fast_params: dict[str, torch.Tensor],
    obs: torch.Tensor,
    actions,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    log_probs_old: torch.Tensor,
    phases,
    action_masks,
    inner_lr: float,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.01,
    value_loss_coef: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Perform one FOMAML inner loop gradient step.

    Computes PPO-style loss via functional_call (no model mutation),
    takes gradient w.r.t. fast_params, returns updated fast_params dict.
    create_graph=False — FOMAML does not differentiate through the inner update.

    Args:
        model: The meta-policy ActorCritic (used as a structure template only).
        fast_params: Current task-adapted parameters (requires_grad=True).
        obs: Observations tensor (T, obs_dim).
        actions: Numpy object array of actions (int or np.array([cat, col])).
        advantages: GAE advantages (T,), already normalised and detached.
        returns: GAE returns (T,), detached.
        log_probs_old: Log-probs from rollout collection (T,).
        phases: Numpy object array of "roll"/"score" strings.
        action_masks: Numpy object array of float32 mask arrays.
        inner_lr: Step size for the gradient update.
        clip_epsilon: PPO clipping epsilon.
        entropy_coef: Entropy bonus coefficient.
        value_loss_coef: Value loss coefficient.

    Returns:
        Updated fast_params dict with same keys as input.
    """
    loss = _ppo_loss_fn(
        model, fast_params, obs, actions, advantages, returns,
        log_probs_old, phases, action_masks,
        clip_epsilon, entropy_coef, value_loss_coef,
    )

    if loss.grad_fn is None:
        return fast_params

    grads = torch.autograd.grad(
        loss,
        list(fast_params.values()),
        create_graph=False,
        allow_unused=True,
    )
    grads = [
        g if g is not None else torch.zeros_like(v)
        for g, v in zip(grads, fast_params.values())
    ]

    # Clip inner-loop gradients to prevent fast_params from exploding
    total_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads))
    clip_coef = 1.0 / (total_norm + 1e-6)
    if clip_coef < 1.0:
        grads = [g * clip_coef for g in grads]

    return {
        k: v - inner_lr * g
        for (k, v), g in zip(fast_params.items(), grads)
    }
