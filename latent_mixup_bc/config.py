"""Frozen experiment configuration and benchmark enumeration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Optional, Tuple


ENVIRONMENTS = ("hopper", "halfcheetah", "walker2d")
QUALITIES = ("medium", "medium-replay", "medium-expert")
METHODS = ("vanilla_bc", "noise_bc", "input_mixup_bc", "latent_mixup_bc")
SEEDS = (0, 1, 2, 3, 4)
MODES = ("SMOKE", "SINGLE_RUN", "FULL_BENCHMARK")


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str = "SMOKE"
    environments: Tuple[str, ...] = ENVIRONMENTS
    qualities: Tuple[str, ...] = QUALITIES
    methods: Tuple[str, ...] = METHODS
    seeds: Tuple[int, ...] = SEEDS
    noise_levels: Tuple[float, ...] = (0.0, 0.01, 0.05, 0.10, 0.20)
    batch_size: int = 256
    learning_rate: float = 3e-4
    latent_dim: int = 128
    mixup_alpha: float = 0.2
    training_noise_std: float = 0.05
    max_epochs: int = 100
    patience: int = 10
    validation_fraction: float = 0.1
    evaluation_episodes: int = 20
    hidden_dim: int = 256
    weight_decay: float = 0.0
    smoke_max_transitions: int = 10_000
    single_dataset: Optional[str] = None
    single_method: Optional[str] = None
    single_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.batch_size <= 0 or self.max_epochs <= 0 or self.evaluation_episodes <= 0:
            raise ValueError("batch_size, max_epochs, and evaluation_episodes must be positive")
        if self.mode == "SINGLE_RUN":
            if self.single_dataset is None or self.single_method is None or self.single_seed is None:
                raise ValueError("SINGLE_RUN requires single_dataset, single_method, and single_seed")


@dataclass(frozen=True)
class RunSpec:
    dataset_id: str
    env_id: str
    method: str
    seed: int


def _dataset_id(environment: str, quality: str) -> str:
    return f"{environment}-{quality}-v2"


def _env_id(environment: str) -> str:
    names = {"hopper": "Hopper-v5", "halfcheetah": "HalfCheetah-v5", "walker2d": "Walker2d-v5"}
    return names[environment]


def enumerate_runs(config: ExperimentConfig):
    if config.mode == "SMOKE":
        dataset_id = _dataset_id(config.environments[0], config.qualities[0])
        return tuple(
            RunSpec(dataset_id, _env_id(config.environments[0]), method, config.seeds[0])
            for method in config.methods
        )
    if config.mode == "SINGLE_RUN":
        dataset_id = str(config.single_dataset)
        environment = dataset_id.split("-")[0]
        return (RunSpec(dataset_id, _env_id(environment), str(config.single_method), int(config.single_seed)),)
    return tuple(
        RunSpec(_dataset_id(environment, quality), _env_id(environment), method, seed)
        for environment in config.environments
        for quality in config.qualities
        for method in config.methods
        for seed in config.seeds
    )


def config_hash(config: ExperimentConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()[:12]
