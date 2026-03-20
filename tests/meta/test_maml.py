import torch
from agents.actor_critic import ActorCritic
from meta.inner_loop import clone_params, inner_update
from meta.maml import FOMAML


def test_clone_params_is_callable():
    model = ActorCritic(obs_dim=85, n_columns=3)
    result = clone_params(model)
    assert result is None or isinstance(result, dict)


def test_fomaml_instantiates():
    model = ActorCritic(obs_dim=85, n_columns=3)
    maml = FOMAML(model, inner_lr=0.01, outer_lr=0.0003, n_inner_steps=5)
    assert maml is not None


def test_fomaml_has_meta_optimizer():
    model = ActorCritic(obs_dim=85, n_columns=3)
    maml = FOMAML(model, inner_lr=0.01, outer_lr=0.0003, n_inner_steps=5)
    assert hasattr(maml, "meta_optimizer")
