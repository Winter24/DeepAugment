import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from latent_mixup_bc.config import ExperimentConfig, RunSpec, config_hash
from latent_mixup_bc.models import BCPolicy
from latent_mixup_bc.persistence import ArtifactStore, restore_missing_artifacts
from latent_mixup_bc.train import cuda_arch_supported, train_one_run


def test_checkpoint_round_trip_preserves_predictions(tmp_path):
    store = ArtifactStore(tmp_path)
    model = BCPolicy(4, 2, 8)
    x = torch.randn(3, 4)
    expected = model(x).detach()
    store.save_checkpoint_atomic("d", "vanilla_bc", 0, {"model": model.state_dict()})
    clone = BCPolicy(4, 2, 8)
    clone.load_state_dict(store.load_checkpoint("d", "vanilla_bc", 0)["model"])
    assert torch.equal(expected, clone(x).detach())


def test_episode_rows_are_deduplicated_by_full_key(tmp_path):
    store = ArtifactStore(tmp_path)
    rows = pd.DataFrame([{
        "dataset_id": "d", "method": "m", "seed": 0,
        "noise_std": 0.0, "episode": 0, "raw_return": 1.0,
    }])
    store.upsert_episode_rows(rows)
    store.upsert_episode_rows(rows.assign(raw_return=2.0))
    saved = pd.read_csv(store.episode_results_path)
    assert len(saved) == 1
    assert saved.raw_return.iloc[0] == 2.0


def test_each_method_can_train_and_write_complete_manifest(tmp_path):
    x = torch.randn(16, 4)
    y = x[:, :2]
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)
    cfg = ExperimentConfig(mode="SMOKE", max_epochs=1, patience=1, hidden_dim=16, latent_dim=8)
    store = ArtifactStore(tmp_path)
    for method in cfg.methods:
        run = RunSpec("hopper-medium-v2", "Hopper-v2", method, 0)
        result = train_one_run(run, cfg, loader, loader, 4, 2, store, device="cpu")
        assert result["status"] == "complete"
        assert store.is_training_complete(run, config_hash(cfg))


def test_restore_does_not_overwrite_newer_working_artifact(tmp_path):
    source, destination = tmp_path / "input", tmp_path / "working"
    (source / "results").mkdir(parents=True)
    (destination / "results").mkdir(parents=True)
    (source / "results" / "summary.csv").write_text("old")
    (destination / "results" / "summary.csv").write_text("new")
    (source / "results" / "episodes.csv").write_text("copied")
    restore_missing_artifacts(source, destination)
    assert (destination / "results" / "summary.csv").read_text() == "new"
    assert (destination / "results" / "episodes.csv").read_text() == "copied"


def test_cuda_architecture_support_check_rejects_p100_for_sm70_build():
    assert not cuda_arch_supported((6, 0), ["sm_70", "sm_75", "sm_80"])
    assert cuda_arch_supported((7, 5), ["sm_70", "sm_75", "sm_80"])
    assert cuda_arch_supported((8, 6), ["sm_70", "sm_75", "sm_80", "sm_86"])
