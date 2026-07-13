"""D4RL dataset locations and explicitly non-standard modern-simulator scoring."""

from __future__ import annotations


DATASET_BASE_URL = "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2"
MODERN_ENVS = {
    "hopper": "Hopper-v5",
    "halfcheetah": "HalfCheetah-v5",
    "walker2d": "Walker2d-v5",
}
REFERENCE_SCORES = {
    "hopper": (-20.272305, 3234.3),
    "halfcheetah": (-280.178953, 12135.0),
    "walker2d": (1.629008, 4592.3),
}
METRIC_LABEL = "modern_simulator_compatibility_score"


def environment_name(dataset_id: str) -> str:
    name = dataset_id.split("-", 1)[0]
    if name not in MODERN_ENVS:
        raise ValueError(f"unsupported locomotion dataset: {dataset_id!r}")
    return name


def modern_env_id(dataset_id: str) -> str:
    return MODERN_ENVS[environment_name(dataset_id)]


def dataset_url(dataset_id: str) -> str:
    if not dataset_id.endswith("-v2"):
        raise ValueError(f"expected a D4RL v2 dataset ID, got {dataset_id!r}")
    environment_name(dataset_id)
    filename = dataset_id[:-3].replace("-", "_") + "-v2.hdf5"
    return f"{DATASET_BASE_URL}/{filename}"


def compatibility_score(dataset_id: str, raw_return: float) -> float:
    minimum, maximum = REFERENCE_SCORES[environment_name(dataset_id)]
    return 100.0 * (float(raw_return) - minimum) / (maximum - minimum)
