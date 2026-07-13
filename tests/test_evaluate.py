import numpy as np
import pytest
import torch

from latent_mixup_bc.data import StateNormalizer
from latent_mixup_bc.evaluate import evaluate_policy, reset_env, step_env, validate_environment


class Box:
    def __init__(self, shape, low=-1.0, high=1.0):
        self.shape = shape
        self.low = np.full(shape, low, dtype=np.float32)
        self.high = np.full(shape, high, dtype=np.float32)


class FiveTupleEnv:
    observation_space = Box((3,), -10, 10)
    action_space = Box((1,))
    spec = type("Spec", (), {"id": "Hopper-v5", "max_episode_steps": 3})()

    def reset(self, seed=None):
        self.t = 0
        return np.zeros(3, dtype=np.float32), {"seed": seed}

    def step(self, action):
        self.t += 1
        return np.ones(3, dtype=np.float32), 2.0, False, self.t >= 2, {}

def test_five_tuple_api_combines_terminated_and_truncated():
    env = FiveTupleEnv()
    obs = reset_env(env, seed=4)
    _, reward, done, _ = step_env(env, np.zeros(1))
    assert obs.shape == (3,)
    assert reward == 2.0
    assert done is False


def test_environment_version_mismatch_is_rejected():
    env = FiveTupleEnv()
    with pytest.raises(ValueError, match="version mismatch"):
        validate_environment(env, "Hopper-v4", 3, 1)


def test_evaluation_returns_episode_rows():
    env = FiveTupleEnv()
    model = torch.nn.Linear(3, 1)
    normalizer = StateNormalizer(np.zeros(3, np.float32), np.ones(3, np.float32))
    rows = evaluate_policy(env, model, normalizer, "hopper-medium-v2", "vanilla_bc", 0, 0.05, 2, "Hopper-v5")
    assert len(rows) == 2
    assert {row["episode"] for row in rows} == {0, 1}
    assert all(row["raw_return"] == 4.0 for row in rows)
    assert all(row["metric_label"] == "modern_simulator_compatibility_score" for row in rows)
