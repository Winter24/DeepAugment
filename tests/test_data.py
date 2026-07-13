import numpy as np
import pytest

from latent_mixup_bc.data import (
    StateNormalizer,
    split_trajectories,
    trajectory_ranges,
    validate_dataset,
)


def fixture_dataset():
    return {
        "observations": np.arange(24, dtype=np.float32).reshape(6, 4),
        "actions": np.arange(12, dtype=np.float32).reshape(6, 2),
        "terminals": np.array([0, 1, 0, 0, 0, 0], dtype=bool),
        "timeouts": np.array([0, 0, 0, 0, 0, 1], dtype=bool),
    }


def test_trajectory_ranges_include_terminal_and_timeout_rows():
    d = fixture_dataset()
    assert trajectory_ranges(d["terminals"], d["timeouts"]) == [(0, 2), (2, 6)]


def test_split_is_disjoint_and_covers_all_rows():
    train, valid = split_trajectories([(0, 2), (2, 6)], valid_fraction=0.5, seed=7)
    assert set(train).isdisjoint(valid)
    assert sorted(train + valid) == list(range(6))


def test_normalizer_uses_only_supplied_training_rows():
    x = np.array([[0.0], [2.0], [1000.0]], dtype=np.float32)
    normalizer = StateNormalizer.fit(x[:2])
    assert np.allclose(normalizer.mean, [1.0])
    assert np.allclose(normalizer.std, [1.0])
    assert np.allclose(normalizer.transform(x[:2]), [[-1.0], [1.0]])


def test_missing_required_key_is_actionable():
    d = fixture_dataset()
    del d["actions"]
    with pytest.raises(ValueError, match="actions"):
        validate_dataset(d)


def test_unterminated_tail_is_kept_as_a_trajectory():
    terminals = np.array([0, 1, 0, 0], dtype=bool)
    timeouts = np.zeros(4, dtype=bool)
    assert trajectory_ranges(terminals, timeouts) == [(0, 2), (2, 4)]
