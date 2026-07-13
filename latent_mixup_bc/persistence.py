"""Atomic artifact persistence and cross-session resume checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd
import torch


EPISODE_KEY = ["dataset_id", "method", "seed", "noise_std", "episode"]


def restore_missing_artifacts(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.exists():
        raise FileNotFoundError(f"resume input root does not exist: {source}")
    copied = []
    for item in source.rglob("*"):
        if not item.is_file() or item.name.endswith(".tmp"):
            continue
        target = destination / item.relative_to(source)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        copied.append(target)
    return copied


class ArtifactStore:
    def __init__(self, root):
        self.root = Path(root)
        self.episode_results_path = self.root / "results" / "episode_results.csv"
        for directory in ("checkpoints", "histories", "manifests", "results", "figures", "cache"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)

    def checkpoint_path(self, dataset_id, method, seed):
        return self.root / "checkpoints" / dataset_id / method / f"seed_{seed}.pth"

    def history_path(self, dataset_id, method, seed):
        return self.root / "histories" / dataset_id / method / f"seed_{seed}.csv"

    def manifest_path(self, dataset_id, method, seed):
        return self.root / "manifests" / dataset_id / method / f"seed_{seed}.json"

    @staticmethod
    def _temp_path(destination: Path):
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent, delete=False)
        handle.close()
        return Path(handle.name)

    def save_checkpoint_atomic(self, dataset_id, method, seed, payload):
        destination = self.checkpoint_path(dataset_id, method, seed)
        temporary = self._temp_path(destination)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def load_checkpoint(self, dataset_id, method, seed, map_location="cpu"):
        return torch.load(
            self.checkpoint_path(dataset_id, method, seed), map_location=map_location,
            weights_only=False,
        )

    def write_csv_atomic(self, frame: pd.DataFrame, destination: Path):
        temporary = self._temp_path(destination)
        try:
            frame.to_csv(temporary, index=False)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def write_json_atomic(self, payload, destination: Path):
        temporary = self._temp_path(destination)
        try:
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def upsert_episode_rows(self, rows: pd.DataFrame):
        if rows.empty:
            return
        missing = [key for key in EPISODE_KEY if key not in rows]
        if missing:
            raise ValueError(f"episode rows are missing key columns: {missing}")
        existing = pd.read_csv(self.episode_results_path) if self.episode_results_path.exists() else pd.DataFrame()
        combined = pd.concat([existing, rows], ignore_index=True)
        combined = combined.drop_duplicates(EPISODE_KEY, keep="last").sort_values(EPISODE_KEY)
        self.write_csv_atomic(combined, self.episode_results_path)

    def is_training_complete(self, run, expected_hash):
        manifest_path = self.manifest_path(run.dataset_id, run.method, run.seed)
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            manifest.get("status") == "complete"
            and manifest.get("config_hash") == expected_hash
            and self.checkpoint_path(run.dataset_id, run.method, run.seed).exists()
            and self.history_path(run.dataset_id, run.method, run.seed).exists()
        )
