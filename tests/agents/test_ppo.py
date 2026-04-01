import torch
from agents.ppo import PPO, compute_gae
from agents.actor_critic import ActorCritic
from agents.rollout_buffer import RolloutBuffer


def test_ppo_instantiates():
    model = ActorCritic(obs_dim=85, n_columns=3)
    ppo = PPO(model)
    assert ppo is not None


def test_rollout_buffer_instantiates():
    buf = RolloutBuffer()
    assert buf is not None


def test_rollout_buffer_has_expected_fields():
    buf = RolloutBuffer()
    assert hasattr(buf, "obs")
    assert hasattr(buf, "actions")
    assert hasattr(buf, "rewards")
    assert hasattr(buf, "dones")
    assert hasattr(buf, "log_probs")
    assert hasattr(buf, "values")


def test_module_level_compute_gae_returns_advantages_and_returns():
    buf = RolloutBuffer()
    # Add two fake transitions
    import numpy as np
    obs = np.zeros(85, dtype=np.float32)
    mask = np.ones(32, dtype=np.float32)
    buf.add(obs, 0, 1.0, False, -0.5, 0.9, "roll", mask)
    buf.add(obs, 31, 2.0, True, -0.3, 0.7, "roll", mask)
    adv, ret = compute_gae(buf, gae_lambda=0.95)
    assert adv.shape == (2,)
    assert ret.shape == (2,)
    assert isinstance(adv, torch.Tensor)
    assert isinstance(ret, torch.Tensor)
