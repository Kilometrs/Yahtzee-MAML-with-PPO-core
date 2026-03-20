from agents.ppo import PPO
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
