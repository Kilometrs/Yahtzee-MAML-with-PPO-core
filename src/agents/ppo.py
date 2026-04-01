"""PPO inner loop optimizer for FOMAML."""

import numpy as np
import torch
import torch.nn.functional as F

from agents.actor_critic import ActorCritic
from agents.rollout_buffer import RolloutBuffer


def compute_gae(
    buffer: "RolloutBuffer",
    gae_lambda: float,
    gamma: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns from a filled RolloutBuffer.

    Args:
        buffer: Filled rollout buffer.
        gae_lambda: Lambda for GAE smoothing.
        gamma: Discount factor.

    Returns:
        (advantages, returns): Both tensors of shape (T,).
    """
    data = buffer.get()
    rewards = torch.tensor(data["rewards"], dtype=torch.float32)
    values = torch.tensor(data["values"], dtype=torch.float32)
    dones = torch.tensor(data["dones"], dtype=torch.float32)
    T = len(rewards)

    advantages = torch.zeros(T)
    last_adv = 0.0

    for t in reversed(range(T)):
        next_value = values[t + 1].item() if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
        advantages[t] = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_adv
        last_adv = advantages[t].item()

    returns = advantages + values
    return advantages, returns


class PPO:
    """Proximal Policy Optimization inner loop.

    Used as the task-specific optimizer within FOMAML. Holds its own Adam
    optimizer for in-place (standard) training. The FOMAML stateless path
    (fast_params) is Phase 2.
    """

    def __init__(
        self,
        actor_critic: ActorCritic,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        gae_lambda: float = 0.95,
        n_epochs: int = 4,
        batch_size: int = 64,
        lr: float = 3e-4,
    ):
        self.actor_critic = actor_critic
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)

    def compute_gae(self, buffer: RolloutBuffer, gamma: float = 0.99) -> torch.Tensor:
        """Compute Generalized Advantage Estimation.

        Args:
            buffer: Filled rollout buffer.
            gamma: Discount factor.

        Returns:
            advantages: Tensor of shape (T,).
        """
        advantages, _ = compute_gae(buffer, self.gae_lambda, gamma)
        return advantages

    def update(
        self,
        buffer: RolloutBuffer,
        fast_params: dict | None = None,
        inner_lr: float | None = None,
    ) -> float:
        """Run PPO update steps (in-place mode only for Phase 1).

        Args:
            buffer: Filled rollout buffer with episode experience.
            fast_params: Must be None in Phase 1.
            inner_lr: Unused in Phase 1.

        Returns:
            mean_loss: Average policy loss over all update epochs.
        """
        if fast_params is not None:
            raise NotImplementedError("FOMAML inner loop — Phase 2")

        advantages = self.compute_gae(buffer)
        data = buffer.get()

        obs = torch.tensor(data["obs"], dtype=torch.float32)
        old_log_probs = torch.tensor(data["log_probs"], dtype=torch.float32)
        values_old = torch.tensor(data["values"], dtype=torch.float32)
        returns = (advantages + values_old).detach()
        advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).detach()

        phases = data["phases"]          # numpy object array of strings
        actions = data["actions"]        # numpy object array (int or np.array([cat, col]))
        action_masks = data["action_masks"]  # numpy object array of float32 arrays

        T = len(advantages)
        total_policy_loss = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            perm = np.random.permutation(T)

            for start in range(0, T, self.batch_size):
                batch_idx = perm[start:start + self.batch_size]

                b_obs = obs[batch_idx]
                b_adv = advantages[batch_idx]
                b_returns = returns[batch_idx]
                b_old_lp = old_log_probs[batch_idx]
                b_phases = phases[batch_idx]
                b_actions = actions[batch_idx]
                b_masks = action_masks[batch_idx]

                # Accumulate loss over both phases within the batch
                batch_policy_loss = torch.tensor(0.0)
                batch_value_loss = torch.tensor(0.0)
                batch_entropy = torch.tensor(0.0)
                batch_count = 0

                for phase_str in ("roll", "score"):
                    pmask_np = b_phases == phase_str
                    if not pmask_np.any():
                        continue
                    pmask = torch.from_numpy(pmask_np)

                    p_obs = b_obs[pmask]
                    p_adv = b_adv[pmask]
                    p_returns = b_returns[pmask]
                    p_old_lp = b_old_lp[pmask]
                    p_actions = b_actions[pmask_np]
                    p_masks_np = b_masks[pmask_np]

                    logits, values_pred = self.actor_critic.forward(p_obs, phase_str)

                    # Stack stored masks into a 2-D tensor
                    mask_tensor = torch.tensor(
                        np.array(list(p_masks_np), dtype=np.float32),
                        dtype=torch.float32,
                    )
                    logits = logits.masked_fill(mask_tensor == 0, float("-inf"))
                    dist = torch.distributions.Categorical(logits=logits)

                    if phase_str == "score":
                        acts_np = np.array(list(p_actions))  # shape (N, 2)
                        flat = (
                            acts_np[:, 0] * self.actor_critic.n_columns + acts_np[:, 1]
                        )
                        act_tensor = torch.tensor(flat, dtype=torch.long)
                    else:
                        act_tensor = torch.tensor(
                            list(p_actions), dtype=torch.long
                        )

                    new_lp = dist.log_prob(act_tensor)
                    entropy = dist.entropy()

                    ratio = torch.exp(new_lp - p_old_lp)
                    clip_adv = torch.clamp(
                        ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                    ) * p_adv
                    policy_loss = -torch.min(ratio * p_adv, clip_adv).sum()
                    value_loss = 0.5 * F.mse_loss(
                        values_pred.squeeze(-1), p_returns, reduction="sum"
                    )

                    batch_policy_loss = batch_policy_loss + policy_loss
                    batch_value_loss = batch_value_loss + value_loss
                    batch_entropy = batch_entropy + entropy.sum()
                    batch_count += int(pmask_np.sum())

                if batch_count == 0:
                    continue

                loss = (
                    batch_policy_loss / batch_count
                    + self.value_loss_coef * batch_value_loss / batch_count
                    - self.entropy_coef * batch_entropy / batch_count
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_policy_loss += (batch_policy_loss / batch_count).item()
                n_updates += 1

        return total_policy_loss / n_updates if n_updates > 0 else 0.0
