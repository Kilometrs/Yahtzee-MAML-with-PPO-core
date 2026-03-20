import numpy as np
from env.yahtzee_env import YahtzeeEnv


def test_env_instantiates_with_default_columns():
    env = YahtzeeEnv()
    assert env.n_columns == 3


def test_obs_space_shape_n3():
    env = YahtzeeEnv(n_columns=3)
    assert env.observation_space.shape == (85,)


def test_obs_space_shape_n6():
    env = YahtzeeEnv(n_columns=6)
    assert env.observation_space.shape == (163,)


def test_roll_action_space_size():
    env = YahtzeeEnv()
    assert env.roll_action_space.n == 32


def test_score_action_space_shape():
    env = YahtzeeEnv(n_columns=3)
    assert list(env.score_action_space.nvec) == [13, 3]
