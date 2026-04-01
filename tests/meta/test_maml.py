import math  # used by meta_update tests (Task 4)
import torch
from agents.actor_critic import ActorCritic
from agents.ppo import compute_gae  # used by inner_update tests (Task 3)
from env.yahtzee_env import YahtzeeEnv
from meta.inner_loop import clone_params, inner_update, collect_episode
from meta.maml import FOMAML
from tasks.reward_tasks import MaxScore  # used by meta_update tests (Task 4)


def test_clone_params_returns_grad_tensors():
    model = ActorCritic(obs_dim=85, n_columns=3)
    result = clone_params(model)
    assert isinstance(result, dict)
    assert len(result) > 0
    assert all(v.requires_grad for v in result.values())
    # Values are clones, not references to original params
    original = dict(model.named_parameters())
    assert all(not result[k].data_ptr() == original[k].data_ptr() for k in result)


def test_fomaml_instantiates():
    model = ActorCritic(obs_dim=85, n_columns=3)
    maml = FOMAML(model, inner_lr=0.01, outer_lr=0.0003, n_inner_steps=5)
    assert maml is not None


def test_fomaml_has_meta_optimizer():
    model = ActorCritic(obs_dim=85, n_columns=3)
    maml = FOMAML(model, inner_lr=0.01, outer_lr=0.0003, n_inner_steps=5)
    assert hasattr(maml, "meta_optimizer")


def test_collect_episode_returns_nonempty_buffer():
    torch.manual_seed(0)
    model = ActorCritic(obs_dim=85, n_columns=3)
    env = YahtzeeEnv(n_columns=3)
    buf = collect_episode(model, env, torch.device("cpu"))
    assert len(buf) > 0


def test_collect_episode_param_swap_restores_model():
    torch.manual_seed(0)
    model = ActorCritic(obs_dim=85, n_columns=3)
    env = YahtzeeEnv(n_columns=3)
    original_val = next(iter(model.parameters())).clone()
    fast_params = {k: v.clone() + 1.0 for k, v in model.named_parameters()}
    collect_episode(model, env, torch.device("cpu"), params=fast_params)
    restored_val = next(iter(model.parameters()))
    assert torch.allclose(restored_val, original_val), \
        "Model params not restored after collect_episode with params"
