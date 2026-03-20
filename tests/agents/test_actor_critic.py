import torch
from agents.actor_critic import ActorCritic


def test_actor_critic_instantiates():
    model = ActorCritic(obs_dim=85, n_columns=3)
    assert isinstance(model, torch.nn.Module)


def test_actor_critic_has_expected_attributes():
    model = ActorCritic(obs_dim=85, n_columns=3)
    assert hasattr(model, "trunk")
    assert hasattr(model, "roll_head")
    assert hasattr(model, "score_head")
    assert hasattr(model, "value_head")
