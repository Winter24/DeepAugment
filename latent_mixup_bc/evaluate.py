"""Environment compatibility checks and clean/OOD policy rollouts."""

from __future__ import annotations

import numpy as np
import torch

from .benchmark import METRIC_LABEL, compatibility_score


def reset_env(env, seed=None):
    result = env.reset(seed=seed)
    observation = result[0] if isinstance(result, tuple) else result
    return np.asarray(observation, dtype=np.float32)


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
    elif len(result) == 4:
        observation, reward, done, info = result
        done = bool(done)
    else:
        raise ValueError(f"unsupported env.step result length: {len(result)}")
    return np.asarray(observation, dtype=np.float32), float(reward), done, info


def validate_environment(env, expected_env_id, state_dim, action_dim):
    actual_id = getattr(getattr(env, "spec", None), "id", None)
    if actual_id != expected_env_id:
        raise ValueError(f"environment version mismatch: expected {expected_env_id}, got {actual_id}")
    if tuple(env.observation_space.shape) != (state_dim,):
        raise ValueError(f"observation dimension mismatch: expected {state_dim}, got {env.observation_space.shape}")
    if tuple(env.action_space.shape) != (action_dim,):
        raise ValueError(f"action dimension mismatch: expected {action_dim}, got {env.action_space.shape}")
    observation = reset_env(env, seed=0)
    if not np.isfinite(observation).all():
        raise ValueError("environment reset returned non-finite observation")


def normalized_score(dataset_id, raw_return):
    return compatibility_score(dataset_id, raw_return)


@torch.no_grad()
def evaluate_policy(env, model, normalizer, dataset_id, method, seed, noise_std, episodes, expected_env_id, device="cpu"):
    state_dim = int(normalizer.mean.shape[0])
    action_dim = int(np.prod(env.action_space.shape))
    validate_environment(env, expected_env_id, state_dim, action_dim)
    device = torch.device(device)
    model.eval().to(device)
    horizon = int(getattr(getattr(env, "spec", None), "max_episode_steps", 1000) or 1000)
    rows = []
    for episode in range(episodes):
        episode_seed = seed * 100_000 + episode
        rng = np.random.default_rng(episode_seed + int(round(noise_std * 1_000_000)))
        observation = reset_env(env, seed=episode_seed)
        total_reward = 0.0
        steps = 0
        done = False
        while not done and steps < horizon:
            normalized = normalizer.transform(observation[None])[0]
            if noise_std > 0:
                normalized = normalized + rng.normal(0.0, noise_std, normalized.shape).astype(np.float32)
            state = torch.from_numpy(normalized).to(device=device, dtype=torch.float32).unsqueeze(0)
            action = model(state).squeeze(0).cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            observation, reward, done, _ = step_env(env, action)
            total_reward += reward
            steps += 1
        rows.append({
            "dataset_id": dataset_id, "method": method, "seed": seed,
            "noise_std": float(noise_std), "episode": episode,
            "raw_return": total_reward, "normalized_score": normalized_score(dataset_id, total_reward),
            "metric_label": METRIC_LABEL, "steps": steps, "env_id": expected_env_id,
        })
    return rows
