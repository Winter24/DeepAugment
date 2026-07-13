"""Seed-aware statistics, tracker generation, and benchmark figures."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


GROUP_KEYS = ["dataset_id", "method", "noise_std"]


def _seed_means(episodes):
    required = GROUP_KEYS + ["seed", "normalized_score"]
    missing = [column for column in required if column not in episodes]
    if missing:
        raise ValueError(f"episode results missing columns: {missing}")
    return episodes.groupby(GROUP_KEYS + ["seed"], as_index=False)["normalized_score"].mean()


def summarize_results(episodes):
    seed_means = _seed_means(episodes)
    grouped = seed_means.groupby(GROUP_KEYS)["normalized_score"]
    summary = grouped.agg(mean_score="mean", std_score="std", seed_count="count").reset_index()
    summary["sem"] = summary["std_score"] / np.sqrt(summary["seed_count"])
    summary["ci95_low"] = summary["mean_score"] - 1.96 * summary["sem"]
    summary["ci95_high"] = summary["mean_score"] + 1.96 * summary["sem"]
    return summary


def paired_bootstrap(left: pd.Series, right: pd.Series, samples=10_000, seed=0):
    matched = left.index.intersection(right.index)
    if len(matched) == 0:
        raise ValueError("paired bootstrap requires at least one matched seed")
    differences = left.loc[matched].to_numpy(float) - right.loc[matched].to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = rng.choice(differences, size=(samples, len(differences)), replace=True).mean(axis=1)
    return {
        "paired_seeds": int(len(differences)),
        "mean_difference": float(differences.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def build_run_tracker(expected, episodes, manifests):
    tracker = expected.copy()
    if episodes.empty:
        means = pd.DataFrame(columns=["dataset_id", "method", "seed", "mean_score", "evaluated_noise_levels"])
    else:
        clean = episodes.loc[episodes["noise_std"].eq(0.0)]
        means = clean.groupby(["dataset_id", "method", "seed"], as_index=False).agg(
            mean_score=("normalized_score", "mean")
        )
        coverage = episodes.groupby(["dataset_id", "method", "seed"])["noise_std"].nunique().rename("evaluated_noise_levels").reset_index()
        means = means.merge(coverage, how="outer")
    tracker = tracker.merge(means, on=["dataset_id", "method", "seed"], how="left")

    def status(row):
        key = (row.dataset_id, row.method, int(row.seed))
        manifest = manifests.get(key)
        if manifest and manifest.get("status") == "failed":
            return "failed"
        trained = bool(manifest and manifest.get("status") == "complete")
        evaluated = pd.notna(row.get("evaluated_noise_levels"))
        if trained and evaluated:
            return "complete" if int(row.evaluated_noise_levels) >= 5 else "partially evaluated"
        if trained:
            return "trained"
        if evaluated:
            return "partially evaluated"
        return "missing"

    tracker["status"] = tracker.apply(status, axis=1)
    return tracker


def render_all_figures(episodes, histories, output_dir):
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_means = _seed_means(episodes)
    clean = seed_means.loc[seed_means.noise_std.eq(0.0)]

    figures = {}
    fig, ax = plt.subplots(figsize=(12, 5))
    datasets = list(dict.fromkeys(clean["dataset_id"]))
    methods = list(dict.fromkeys(clean["method"]))
    x = np.arange(len(datasets), dtype=float)
    width = 0.8 / max(1, len(methods))
    for method_index, method in enumerate(methods):
        subset = clean.loc[clean.method.eq(method)]
        grouped = subset.groupby("dataset_id")["normalized_score"]
        means = grouped.mean().reindex(datasets)
        errors = grouped.std().reindex(datasets).fillna(0.0)
        positions = x - 0.4 + width / 2 + method_index * width
        ax.bar(positions, means, width, yerr=errors, capsize=3, label=method)
    ax.set_xticks(x, datasets, rotation=45, ha="right")
    ax.set_ylabel("Modern simulator compatibility score")
    ax.legend()
    figures["clean_scores"] = fig

    fig, ax = plt.subplots(figsize=(9, 5))
    for method, subset in seed_means.groupby("method"):
        grouped = subset.groupby("noise_std")["normalized_score"]
        means, errors = grouped.mean(), grouped.std().fillna(0.0)
        ax.errorbar(means.index.to_numpy(), means.to_numpy(), yerr=errors.to_numpy(), marker="o", capsize=3, label=method)
    ax.set_xlabel("Observation noise std (normalized units)")
    ax.set_ylabel("Modern simulator compatibility score")
    ax.legend()
    figures["robustness"] = fig

    clean_keys = seed_means.loc[seed_means.noise_std.eq(0.0), ["dataset_id", "method", "seed", "normalized_score"]].rename(columns={"normalized_score": "clean_score"})
    drop = seed_means.merge(clean_keys, on=["dataset_id", "method", "seed"], how="left")
    drop["relative_drop"] = (drop["clean_score"] - drop["normalized_score"]) / drop["clean_score"].abs().clip(lower=1e-8)
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, subset in drop.groupby("method"):
        grouped = subset.groupby("noise_std")["relative_drop"]
        means, errors = grouped.mean(), grouped.std().fillna(0.0)
        ax.errorbar(means.index.to_numpy(), means.to_numpy(), yerr=errors.to_numpy(), marker="o", capsize=3, label=method)
    ax.set_xlabel("Observation noise std (normalized units)")
    ax.set_ylabel("Relative drop from clean score")
    ax.legend()
    figures["dropoff"] = fig

    if histories is not None and not histories.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for method, subset in histories.groupby("method"):
            grouped = subset.groupby("epoch")["valid_mse"].mean()
            ax.plot(grouped.index.to_numpy(), grouped.to_numpy(), label=method)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MSE")
        ax.legend()
        figures["validation_loss"] = fig

    paths = []
    for name, fig in figures.items():
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = output_dir / f"{name}.{suffix}"
            fig.savefig(path, dpi=180)
            paths.append(path)
        plt.close(fig)
    return paths
