from latent_mixup_bc.config import ExperimentConfig, config_hash, enumerate_runs


def test_full_benchmark_has_180_unique_runs():
    cfg = ExperimentConfig(mode="FULL_BENCHMARK")
    runs = enumerate_runs(cfg)
    keys = {(r.dataset_id, r.method, r.seed) for r in runs}
    assert len(runs) == 180
    assert len(keys) == 180


def test_hash_is_stable_and_changes_with_training_semantics():
    a = ExperimentConfig(mode="SMOKE", learning_rate=3e-4)
    b = ExperimentConfig(mode="SMOKE", learning_rate=3e-4)
    c = ExperimentConfig(mode="SMOKE", learning_rate=1e-3)
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)


def test_smoke_enumerates_all_methods_for_one_dataset_and_seed():
    runs = enumerate_runs(ExperimentConfig(mode="SMOKE"))
    assert len(runs) == 4
    assert {r.method for r in runs} == {
        "vanilla_bc", "noise_bc", "input_mixup_bc", "latent_mixup_bc"
    }
    assert {r.env_id for r in runs} == {"Hopper-v5"}
