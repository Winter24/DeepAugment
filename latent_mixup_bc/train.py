"""Deterministic training and early stopping for the four BC methods."""

from __future__ import annotations

from dataclasses import asdict
import random
import re
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import config_hash
from .models import BCPolicy, prepare_training_batch


def cuda_arch_supported(capability, arch_list):
    supported = []
    for architecture in arch_list:
        match = re.fullmatch(r"(?:sm|compute)_(\d+)", architecture)
        if match:
            supported.append(int(match.group(1)))
    if not supported:
        return False
    device_arch = int(capability[0]) * 10 + int(capability[1])
    return min(supported) <= device_arch <= max(supported)


def select_device(requested=None):
    if requested is not None and str(requested).startswith("cpu"):
        return torch.device("cpu")
    wants_cuda = requested is None or str(requested).startswith("cuda")
    if wants_cuda and torch.cuda.is_available():
        try:
            capability = torch.cuda.get_device_capability(0)
            architectures = torch.cuda.get_arch_list()
            if cuda_arch_supported(capability, architectures):
                return torch.device(requested or "cuda")
            warnings.warn(
                f"GPU capability sm_{capability[0]}{capability[1]} is not supported by this "
                f"PyTorch build ({architectures}); falling back to CPU. Select a Kaggle T4 "
                "accelerator for GPU training with the current image.",
                RuntimeWarning,
            )
        except Exception as exc:
            warnings.warn(f"CUDA compatibility check failed ({exc}); falling back to CPU.", RuntimeWarning)
    return torch.device("cpu")


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _training_prediction(model, method, states, actions, config):
    if method == "latent_mixup_bc":
        latent = model.encode(states)
        batch = prepare_training_batch(
            method, latent, actions, alpha=config.mixup_alpha, representation_is_latent=True
        )
        return model.act_from_latent(batch.representation), batch.target
    batch = prepare_training_batch(
        method, states, actions, alpha=config.mixup_alpha, noise_std=config.training_noise_std
    )
    return model(batch.representation), batch.target


@torch.no_grad()
def validation_mse(model, loader, device):
    criterion = nn.MSELoss(reduction="sum")
    loss_sum = 0.0
    element_count = 0
    model.eval()
    for states, actions in loader:
        states, actions = states.to(device), actions.to(device)
        prediction = model(states)
        loss_sum += criterion(prediction, actions).item()
        element_count += actions.numel()
    return loss_sum / max(1, element_count)


def train_one_run(run, config, train_loader, valid_loader, state_dim, action_dim, store, device=None, normalizer=None):
    run_hash = config_hash(config)
    if store.is_training_complete(run, run_hash):
        return {"status": "complete", "skipped": True}
    device = select_device(device)
    set_global_seed(run.seed)
    model = BCPolicy(state_dim, action_dim, config.latent_dim, config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0
    history = []
    max_epochs = 2 if config.mode == "SMOKE" else config.max_epochs

    for epoch in range(max_epochs):
        model.train()
        loss_sum = 0.0
        batches = 0
        for states, actions in train_loader:
            states, actions = states.to(device), actions.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, target = _training_prediction(model, run.method, states, actions, config)
            loss = criterion(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss for {run}")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            batches += 1
        valid_loss = validation_mse(model, valid_loader, device)
        history.append({"epoch": epoch + 1, "train_mse": loss_sum / max(1, batches), "valid_mse": valid_loss})
        if valid_loss < best_loss:
            best_loss, best_epoch, stale_epochs = valid_loss, epoch + 1, 0
            store.save_checkpoint_atomic(run.dataset_id, run.method, run.seed, {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": asdict(config), "config_hash": run_hash,
                "state_dim": state_dim, "action_dim": action_dim,
                "best_epoch": best_epoch, "best_valid_mse": best_loss,
                "normalizer": normalizer.to_dict() if normalizer is not None else None,
            })
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    history_path = store.history_path(run.dataset_id, run.method, run.seed)
    store.write_csv_atomic(pd.DataFrame(history), history_path)
    manifest_path = store.manifest_path(run.dataset_id, run.method, run.seed)
    store.write_json_atomic({
        "status": "complete", "config_hash": run_hash, "dataset_id": run.dataset_id,
        "env_id": run.env_id, "method": run.method, "seed": run.seed,
        "best_epoch": best_epoch, "best_valid_mse": best_loss,
        "checkpoint": str(store.checkpoint_path(run.dataset_id, run.method, run.seed)),
        "history": str(history_path),
    }, manifest_path)
    return {"status": "complete", "skipped": False, "best_epoch": best_epoch, "best_valid_mse": best_loss}


def load_best_policy(run, config, store, device="cpu"):
    payload = store.load_checkpoint(run.dataset_id, run.method, run.seed, map_location=device)
    if payload.get("config_hash") != config_hash(config):
        raise ValueError("checkpoint configuration hash does not match current experiment")
    model = BCPolicy(payload["state_dim"], payload["action_dim"], config.latent_dim, config.hidden_dim)
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), payload
