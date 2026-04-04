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


def preprocess_rollout_for_device(sd: dict, device: torch.device, n_columns: int) -> dict:
    """Pre-process rollout data into phase-separated GPU tensors.

    Converts numpy object arrays into properly typed tensors ONCE,
    avoiding repeated list() and np.array() conversions in _ppo_loss_fn.

    Returns dict with keys: roll, score — each containing pre-processed tensors.
    """
    phases = sd["phases"]
    result = {}

    for phase_str in ("roll", "score"):
        pmask_np = phases == phase_str
        if not pmask_np.any():
            result[phase_str] = None
            continue

        p_obs = torch.tensor(sd["obs"][pmask_np], device=device)
        p_log_probs = torch.tensor(sd["log_probs"][pmask_np], device=device)

        # Pre-stack action masks into a proper 2D float tensor
        masks_list = sd["action_masks"][pmask_np]
        mask_tensor = torch.tensor(
            np.stack(masks_list), dtype=torch.float32, device=device
        )

        # Pre-convert actions to flat long tensor
        actions_list = sd["actions"][pmask_np]
        if phase_str == "score":
            acts_np = np.stack(actions_list)  # (N, 2)
            act_tensor = torch.tensor(
                acts_np[:, 0] * n_columns + acts_np[:, 1],
                dtype=torch.long, device=device,
            )
        else:
            act_tensor = torch.tensor(
                np.array(list(actions_list), dtype=np.int64),
                dtype=torch.long, device=device,
            )

        # Boolean index tensor for selecting from full advantage/return arrays
        pmask_torch = torch.from_numpy(pmask_np).to(device)

        result[phase_str] = {
            "obs": p_obs,
            "log_probs": p_log_probs,
            "mask": mask_tensor,
            "actions": act_tensor,
            "pmask": pmask_torch,
            "count": int(pmask_np.sum()),
        }

    return result


class _TracedInference(torch.nn.Module):
    """Wraps ActorCritic for JIT tracing — computes all heads in one pass."""
    def __init__(self, model: ActorCritic):
        super().__init__()
        self.trunk = model.trunk
        self.roll_head = model.roll_head
        self.score_head = model.score_head
        self.value_head = model.value_head

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        return self.roll_head(features), self.score_head(features), self.value_head(features)


# Per-worker cached traced model (avoids re-tracing on every episode)
_worker_traced_cache: dict = {}


def _get_or_build_traced(model_kwargs: dict, state: dict) -> torch.jit.ScriptModule:
    """Get cached traced model or build + cache one. Updates weights if cached."""
    cache_key = (model_kwargs["obs_dim"], model_kwargs["n_columns"],
                 model_kwargs["hidden_dim"], model_kwargs["n_layers"])

    if cache_key in _worker_traced_cache:
        traced, underlying = _worker_traced_cache[cache_key]
        # Fast weight update — load_state_dict on the underlying model,
        # traced model shares the same parameters
        underlying.load_state_dict(state)
        return traced

    model = ActorCritic(**model_kwargs)
    model.load_state_dict(state)
    model.eval()
    wrapper = _TracedInference(model)
    wrapper.eval()
    dummy = torch.randn(1, model_kwargs["obs_dim"])
    traced = torch.jit.trace(wrapper, dummy)
    _worker_traced_cache[cache_key] = (traced, model)
    return traced


def collect_episode_cpu(
    model_state: dict,
    model_kwargs: dict,
    task,
    n_columns: int,
    fast_params_state: dict | None = None,
) -> dict:
    """Collect one episode entirely on CPU. Safe to call in a worker process.

    Uses JIT-traced model with per-worker caching for fast inference.

    Args:
        model_state: state_dict of the meta-model (CPU tensors).
        model_kwargs: kwargs to reconstruct ActorCritic (obs_dim, n_columns, etc).
        task: BaseTask instance with a .reward method (picklable).
        n_columns: Used to construct YahtzeeEnv inside the worker.
        fast_params_state: If provided, load these weights instead of model_state
                           (used for query rollout collection with adapted params).

    Returns:
        dict from RolloutBuffer.get() — plain numpy arrays, picklable.
    """
    from env.yahtzee_env import YahtzeeEnv

    # Limit intra-op threads to avoid contention when running in parallel workers
    torch.set_num_threads(2)

    weights = fast_params_state if fast_params_state is not None else model_state
    traced = _get_or_build_traced(model_kwargs, weights)
    n_cols = model_kwargs["n_columns"]

    buf = RolloutBuffer()
    env = YahtzeeEnv(n_columns=n_columns)
    env.reward_fn = task.reward
    obs, info = env.reset()

    while True:
        phase = info["phase"]
        mask = info["action_mask"]

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            roll_logits, score_logits, value = traced(obs_t)

        # Select logits based on phase and apply mask
        logits = roll_logits if phase == "roll" else score_logits
        mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        logits = logits.masked_fill(mask_t == 0, float("-inf"))

        dist = torch.distributions.Categorical(logits=logits)
        idx = dist.sample()
        log_prob = dist.log_prob(idx)

        idx_val = idx.squeeze(0)
        log_prob_val = log_prob.squeeze(0)

        if phase == "score":
            cat = (idx_val // n_cols).item()
            col = (idx_val % n_cols).item()
            action = np.array([cat, col])
        else:
            action = idx_val.item()

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated

        buf.add(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            log_prob=log_prob_val.item(),
            value=value.squeeze().item(),
            phase=phase,
            action_mask=mask,
        )

        if done:
            break

        obs = next_obs
        info = next_info

    return buf.get()


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


def _manual_forward(fast_params: dict[str, torch.Tensor], obs: torch.Tensor, n_layers: int):
    """Forward pass using direct matmul on fast_params — no functional_call overhead.

    Computes trunk + all three heads in one pass (roll, score, value).
    ~40% faster than functional_call for the same computation.
    """
    x = obs
    for i in range(n_layers):
        x = F.relu(F.linear(x, fast_params[f"trunk.{i*2}.weight"], fast_params[f"trunk.{i*2}.bias"]))
    roll_logits = F.linear(x, fast_params["roll_head.weight"], fast_params["roll_head.bias"])
    score_logits = F.linear(x, fast_params["score_head.weight"], fast_params["score_head.bias"])
    value = F.linear(x, fast_params["value_head.weight"], fast_params["value_head.bias"])
    return roll_logits, score_logits, value


def _ppo_loss_fn_fast(
    model: nn.Module,
    fast_params: dict[str, torch.Tensor],
    preprocessed: dict,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_epsilon: float,
    entropy_coef: float,
    value_loss_coef: float,
) -> torch.Tensor:
    """Fast PPO loss using pre-processed phase-separated tensors.

    Uses direct matmul instead of functional_call for ~40% faster forward pass.
    Processes all observations through trunk once, then splits by phase for head selection.
    """
    n_layers = len([k for k in fast_params if k.startswith("trunk.") and k.endswith(".weight")])

    # Concatenate all observations and run trunk once
    obs_parts = []
    split_sizes = []
    phase_order = []
    for phase_str in ("roll", "score"):
        pd = preprocessed.get(phase_str)
        if pd is not None:
            obs_parts.append(pd["obs"])
            split_sizes.append(pd["obs"].shape[0])
            phase_order.append(phase_str)

    if not obs_parts:
        return torch.tensor(0.0)

    all_obs = torch.cat(obs_parts, dim=0) if len(obs_parts) > 1 else obs_parts[0]

    # Single trunk forward pass for ALL steps
    x = all_obs
    for i in range(n_layers):
        x = F.relu(F.linear(x, fast_params[f"trunk.{i*2}.weight"], fast_params[f"trunk.{i*2}.bias"]))

    # Compute all heads on shared features
    all_roll = F.linear(x, fast_params["roll_head.weight"], fast_params["roll_head.bias"])
    all_score = F.linear(x, fast_params["score_head.weight"], fast_params["score_head.bias"])
    all_value = F.linear(x, fast_params["value_head.weight"], fast_params["value_head.bias"])

    # Split back by phase
    if len(split_sizes) > 1:
        roll_splits = torch.split(all_roll, split_sizes)
        score_splits = torch.split(all_score, split_sizes)
        value_splits = torch.split(all_value, split_sizes)
    else:
        roll_splits = [all_roll]
        score_splits = [all_score]
        value_splits = [all_value]

    loss_terms = []
    total_count = 0

    for idx, phase_str in enumerate(phase_order):
        pd = preprocessed[phase_str]
        p_adv = advantages[pd["pmask"]]
        p_returns = returns[pd["pmask"]]
        p_old_lp = pd["log_probs"]

        logits = roll_splits[idx] if phase_str == "roll" else score_splits[idx]
        values_pred = value_splits[idx]

        logits = logits.masked_fill(pd["mask"] == 0, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)

        new_lp = dist.log_prob(pd["actions"])
        entropy = dist.entropy()

        ratio = torch.exp(new_lp - p_old_lp)
        clip_adv = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * p_adv
        policy_loss = -torch.min(ratio * p_adv, clip_adv).sum()
        value_loss = 0.5 * F.mse_loss(
            values_pred.squeeze(-1), p_returns, reduction="sum"
        )
        entropy_loss = -entropy_coef * entropy.sum()

        loss_terms.append(policy_loss + value_loss_coef * value_loss + entropy_loss)
        total_count += pd["count"]

    return torch.stack(loss_terms).sum() / total_count


def _grad_step(fast_params, loss, inner_lr):
    """Shared gradient step logic for inner updates."""
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

    total_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads))
    clip_coef = 1.0 / (total_norm + 1e-6)
    if clip_coef < 1.0:
        grads = [g * clip_coef for g in grads]

    return {
        k: v - inner_lr * g
        for (k, v), g in zip(fast_params.items(), grads)
    }


def inner_update_fast(
    model: nn.Module,
    fast_params: dict[str, torch.Tensor],
    preprocessed: dict,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    inner_lr: float,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.01,
    value_loss_coef: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Fast inner update using pre-processed tensors."""
    loss = _ppo_loss_fn_fast(
        model, fast_params, preprocessed, advantages, returns,
        clip_epsilon, entropy_coef, value_loss_coef,
    )
    return _grad_step(fast_params, loss, inner_lr)


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
    return _grad_step(fast_params, loss, inner_lr)
