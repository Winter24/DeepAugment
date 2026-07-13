import pandas as pd

from latent_mixup_bc.report import (
    build_run_tracker,
    paired_bootstrap,
    render_all_figures,
    summarize_results,
)


def test_missing_run_stays_missing_not_zero():
    expected = pd.DataFrame([
        {"dataset_id": "d", "method": "a", "seed": 0},
        {"dataset_id": "d", "method": "b", "seed": 0},
    ])
    episodes = pd.DataFrame([{
        "dataset_id": "d", "method": "a", "seed": 0, "noise_std": 0.0,
        "episode": 0, "normalized_score": 10.0,
    }])
    tracker = build_run_tracker(expected, episodes, manifests={})
    missing = tracker.loc[tracker.method.eq("b")].iloc[0]
    assert missing.status == "missing"
    assert pd.isna(missing.mean_score)


def test_summary_reports_seed_count_after_episode_aggregation():
    df = pd.DataFrame({
        "dataset_id": ["d"] * 4, "method": ["a"] * 4,
        "seed": [0, 0, 1, 1], "noise_std": [0.0] * 4,
        "episode": [0, 1, 0, 1], "normalized_score": [1.0, 3.0, 3.0, 5.0],
    })
    out = summarize_results(df)
    assert out.seed_count.iloc[0] == 2
    assert out.mean_score.iloc[0] == 3.0


def test_paired_bootstrap_uses_only_matched_seeds():
    left = pd.Series({0: 3.0, 1: 5.0})
    right = pd.Series({0: 1.0, 1: 3.0, 2: 100.0})
    result = paired_bootstrap(left, right, samples=100, seed=0)
    assert result["paired_seeds"] == 2
    assert result["mean_difference"] == 2.0


def test_required_figures_are_written(tmp_path):
    rows = []
    for method_index, method in enumerate(("vanilla_bc", "latent_mixup_bc")):
        for seed in (0, 1):
            for noise in (0.0, 0.1):
                rows.append({
                    "dataset_id": "hopper-medium-v2", "method": method,
                    "seed": seed, "noise_std": noise, "episode": 0,
                    "normalized_score": 50 - method_index - 10 * noise,
                })
    histories = pd.DataFrame({
        "epoch": [1, 2, 1, 2], "valid_mse": [2.0, 1.0, 2.2, 1.1],
        "method": ["vanilla_bc", "vanilla_bc", "latent_mixup_bc", "latent_mixup_bc"],
    })
    paths = render_all_figures(pd.DataFrame(rows), histories, tmp_path)
    assert {path.stem for path in paths} == {
        "clean_scores", "robustness", "dropoff", "validation_loss"
    }
    assert all(path.stat().st_size > 0 for path in paths)
