"""Validated D4RL loading and leakage-safe preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


REQUIRED_KEYS = ("observations", "actions", "terminals", "timeouts")


def validate_dataset(dataset: Mapping[str, np.ndarray]) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in dataset]
    if missing:
        raise ValueError(f"D4RL dataset is missing required keys: {missing}")
    arrays = {key: np.asarray(dataset[key]) for key in REQUIRED_KEYS}
    n = len(arrays["observations"])
    if n == 0:
        raise ValueError("D4RL dataset contains no transitions")
    if arrays["observations"].ndim != 2 or arrays["actions"].ndim != 2:
        raise ValueError("observations and actions must both be rank-2 arrays")
    bad_lengths = {key: len(value) for key, value in arrays.items() if len(value) != n}
    if bad_lengths:
        raise ValueError(f"D4RL arrays have inconsistent lengths: expected {n}, got {bad_lengths}")
    if not np.isfinite(arrays["observations"]).all():
        raise ValueError("observations contain NaN or infinite values")
    if not np.isfinite(arrays["actions"]).all():
        raise ValueError("actions contain NaN or infinite values")


def trajectory_ranges(terminals: np.ndarray, timeouts: np.ndarray):
    terminals = np.asarray(terminals, dtype=bool).reshape(-1)
    timeouts = np.asarray(timeouts, dtype=bool).reshape(-1)
    if len(terminals) != len(timeouts):
        raise ValueError("terminals and timeouts must have equal lengths")
    ranges = []
    start = 0
    for index, ended in enumerate(terminals | timeouts):
        if ended:
            ranges.append((start, index + 1))
            start = index + 1
    if start < len(terminals):
        ranges.append((start, len(terminals)))
    return ranges


def split_trajectories(
    ranges: Sequence[tuple[int, int]], valid_fraction: float, seed: int
):
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be between 0 and 1")
    if len(ranges) < 2:
        raise ValueError("at least two trajectories are required for a disjoint split")
    order = np.random.default_rng(seed).permutation(len(ranges))
    n_valid = min(len(ranges) - 1, max(1, int(round(len(ranges) * valid_fraction))))
    valid_trajectories = set(order[:n_valid].tolist())
    train_rows, valid_rows = [], []
    for index, (start, stop) in enumerate(ranges):
        target = valid_rows if index in valid_trajectories else train_rows
        target.extend(range(start, stop))
    return train_rows, valid_rows


@dataclass(frozen=True)
class StateNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, states: np.ndarray) -> "StateNormalizer":
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2 or len(states) == 0:
            raise ValueError("training states must be a non-empty rank-2 array")
        mean = states.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = states.std(axis=0, dtype=np.float64).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return cls(mean=mean, std=std)

    def transform(self, states: np.ndarray) -> np.ndarray:
        return ((np.asarray(states, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload):
        return cls(np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["std"], dtype=np.float32))


def load_d4rl_hdf5(path: str | Path):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to read cached D4RL datasets") from exc
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"D4RL HDF5 file not found: {path}")
    with h5py.File(path, "r") as handle:
        dataset = {key: np.asarray(handle[key]) for key in REQUIRED_KEYS if key in handle}
    validate_dataset(dataset)
    return dataset


def make_loaders(
    dataset: Mapping[str, np.ndarray], batch_size: int, valid_fraction: float, seed: int,
    max_transitions: int | None = None,
):
    validate_dataset(dataset)
    ranges = trajectory_ranges(dataset["terminals"], dataset["timeouts"])
    train_rows, valid_rows = split_trajectories(ranges, valid_fraction, seed)
    if max_transitions is not None:
        train_rows = train_rows[:max_transitions]
        valid_rows = valid_rows[:max(1, max_transitions // 10)]
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    actions = np.asarray(dataset["actions"], dtype=np.float32)
    normalizer = StateNormalizer.fit(observations[train_rows])

    def loader(rows, shuffle):
        states = torch.from_numpy(normalizer.transform(observations[rows]))
        targets = torch.from_numpy(actions[rows].astype(np.float32))
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            TensorDataset(states, targets), batch_size=batch_size, shuffle=shuffle,
            generator=generator if shuffle else None,
        )

    return loader(train_rows, True), loader(valid_rows, False), normalizer, train_rows, valid_rows
