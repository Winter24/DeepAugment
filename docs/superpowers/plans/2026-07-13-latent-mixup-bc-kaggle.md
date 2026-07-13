# Latent Mixup BC Kaggle Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the prototype notebook with a self-contained, resumable Kaggle benchmark for four controlled Behavior Cloning augmentation methods on nine D4RL MuJoCo datasets.

**Architecture:** Testable Python modules are the canonical implementation; a deterministic builder embeds those modules into one standalone Kaggle notebook with management Markdown before every executable section. Training data and simulator access use separate adapters, while all durable state is keyed by dataset, method, seed, and configuration hash.

**Tech Stack:** Python 3, PyTorch, NumPy, pandas, h5py, matplotlib, seaborn, Gym/D4RL MuJoCo compatibility backend, pytest, nbformat.

## Global Constraints

- Methods are exactly `vanilla_bc`, `noise_bc`, `input_mixup_bc`, and `latent_mixup_bc`.
- Full benchmark is 3 environments × 3 dataset qualities × 4 methods × 5 seeds = 180 training runs.
- D4RL data may never silently fall back to synthetic data.
- Dataset `v2` policies may not be reported as standard D4RL results when evaluated in Gymnasium `v4` or `v5` environments.
- State statistics come only from training trajectories.
- Mixup representation and target always share one coefficient tensor and one permutation.
- Missing runs are missing data, never zero scores.
- Kaggle execution supports `SMOKE`, `SINGLE_RUN`, and `FULL_BENCHMARK` plus cross-session resume.
- No benchmark values may be fabricated.

---

## File Structure

- Create `latent_mixup_bc/__init__.py`: stable public exports.
- Create `latent_mixup_bc/config.py`: frozen experiment configuration, run enumeration, and hashes.
- Create `latent_mixup_bc/data.py`: D4RL HDF5 loading, trajectory reconstruction/splitting, normalization, DataLoaders.
- Create `latent_mixup_bc/models.py`: controlled MLP and four training-batch transformations.
- Create `latent_mixup_bc/persistence.py`: atomic checkpoints/CSV/manifests and resume state.
- Create `latent_mixup_bc/train.py`: deterministic training, validation, and early stopping.
- Create `latent_mixup_bc/evaluate.py`: simulator API adapter, clean/OOD rollout, normalized scores.
- Create `latent_mixup_bc/report.py`: aggregation, bootstrap intervals, tracker, tables, and figures.
- Create `scripts/build_kaggle_notebook.py`: deterministic standalone notebook assembly.
- Replace `Latent State-Action Mixup.ipynb`: generated standalone Kaggle artifact.
- Create `tests/`: focused unit and integration tests for each module and notebook structure.
- Create `README.md`, `docs/EXPERIMENT_PROTOCOL.md`, `docs/KAGGLE_RUNBOOK.md`, `docs/RUN_TRACKER.md`, and `docs/RESULTS_TEMPLATE.md`.

### Task 1: Frozen configuration and exact benchmark enumeration

**Files:**
- Create: `latent_mixup_bc/__init__.py`
- Create: `latent_mixup_bc/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `ExperimentConfig`, `RunSpec`, `enumerate_runs(config)`, `config_hash(config)`.
- Consumers: scheduler, artifact paths, notebook dashboard, tracker, and all later tests.

- [ ] **Step 1: Write failing configuration tests**

```python
from latent_mixup_bc.config import ExperimentConfig, enumerate_runs, config_hash


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
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_config.py -q`

Expected: collection fails because `latent_mixup_bc.config` does not exist.

- [ ] **Step 3: Implement immutable configuration and enumeration**

Implement frozen dataclasses with these defaults: environments `hopper`, `halfcheetah`, `walker2d`; qualities `medium`, `medium-replay`, `medium-expert`; methods named above; seeds `(0,1,2,3,4)`; noise levels `(0.0,0.01,0.05,0.10,0.20)`; batch size 256; learning rate `3e-4`; latent dimension 128; mixup alpha 0.2; training noise 0.05; maximum epochs 100; patience 10; validation fraction 0.1; 20 evaluation episodes. Serialize all training/evaluation semantics using sorted JSON and hash with SHA-256 truncated to 12 characters.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_config.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Optional Git checkpoint**

If the workspace is initialized as a valid repository: `git add latent_mixup_bc tests/test_config.py && git commit -m "feat: freeze benchmark configuration"`.

### Task 2: D4RL data adapter, trajectory split, and normalization

**Files:**
- Create: `latent_mixup_bc/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, D4RL mapping with `observations`, `actions`, `terminals`, `timeouts`.
- Produces: `validate_dataset`, `trajectory_ranges`, `split_trajectories`, `StateNormalizer`, `make_loaders`, `load_d4rl_hdf5`.

- [ ] **Step 1: Write failing tests for validation, trajectory boundaries, leakage, and train-only statistics**

```python
import numpy as np
import pytest
from latent_mixup_bc.data import (
    StateNormalizer, split_trajectories, trajectory_ranges, validate_dataset,
)


def fixture_dataset():
    return {
        "observations": np.arange(24, dtype=np.float32).reshape(6, 4),
        "actions": np.arange(12, dtype=np.float32).reshape(6, 2),
        "terminals": np.array([0, 1, 0, 0, 0, 0], dtype=bool),
        "timeouts": np.array([0, 0, 0, 0, 0, 1], dtype=bool),
    }


def test_trajectory_ranges_include_terminal_and_timeout_rows():
    d = fixture_dataset()
    assert trajectory_ranges(d["terminals"], d["timeouts"]) == [(0, 2), (2, 6)]


def test_split_is_disjoint_and_covers_all_rows():
    train, valid = split_trajectories([(0, 2), (2, 6)], valid_fraction=0.5, seed=7)
    assert set(train).isdisjoint(valid)
    assert sorted(train + valid) == list(range(6))


def test_normalizer_uses_only_supplied_training_rows():
    x = np.array([[0.0], [2.0], [1000.0]], dtype=np.float32)
    n = StateNormalizer.fit(x[:2])
    assert np.allclose(n.mean, [1.0])
    assert np.allclose(n.std, [1.0])


def test_missing_required_key_is_actionable():
    d = fixture_dataset()
    del d["actions"]
    with pytest.raises(ValueError, match="actions"):
        validate_dataset(d)
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_data.py -q`

Expected: import failure for missing data module.

- [ ] **Step 3: Implement the minimal validated loader pipeline**

Load HDF5 arrays with `h5py`, reject non-finite/shape-inconsistent arrays, treat either terminal or timeout as a trajectory end, append a final unterminated range if necessary, split whole trajectories deterministically, clamp normalization standard deviations below `1e-6` to `1.0`, and return float32 tensors. Do not include any synthetic-data function.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_data.py -q`

Expected: `4 passed`.

### Task 3: Controlled policy and augmentation semantics

**Files:**
- Create: `latent_mixup_bc/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `BCPolicy`, `MixupBatch`, `sample_mixup`, `prepare_training_batch`.
- Consumers: trainer and notebook self-tests.

- [ ] **Step 1: Write failing tests that expose the old permutation bug**

```python
import torch
from latent_mixup_bc.models import BCPolicy, prepare_training_batch


def test_input_mixup_reuses_lambda_and_permutation_for_actions():
    s = torch.tensor([[0.0], [10.0]])
    a = torch.tensor([[1.0], [3.0]])
    lam = torch.tensor([0.25, 0.75])
    perm = torch.tensor([1, 0])
    batch = prepare_training_batch("input_mixup_bc", s, a, lam=lam, permutation=perm)
    assert torch.allclose(batch.representation, lam[:, None] * s + (1-lam[:, None]) * s[perm])
    assert torch.allclose(batch.target, lam[:, None] * a + (1-lam[:, None]) * a[perm])
    assert torch.equal(batch.permutation, perm)


def test_every_method_has_the_same_policy_shape():
    model = BCPolicy(11, 3, latent_dim=128)
    assert model(torch.zeros(7, 11)).shape == (7, 3)
    assert model.encode(torch.zeros(7, 11)).shape == (7, 128)
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_models.py -q`

Expected: import failure for missing model module.

- [ ] **Step 3: Implement one common architecture and explicit transformations**

`MixupBatch` carries `representation`, `target`, `lambda_`, and `permutation`. `sample_mixup` samples beta coefficients and exactly one permutation. `prepare_training_batch` handles vanilla, training noise, and input mixup; latent mixup is applied after `model.encode` with the same `MixupBatch` metadata. Unknown method names raise `ValueError`.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_models.py -q`

Expected: `2 passed`.

### Task 4: Training and durable resume state

**Files:**
- Create: `latent_mixup_bc/persistence.py`
- Create: `latent_mixup_bc/train.py`
- Test: `tests/test_train_persistence.py`

**Interfaces:**
- Produces: `ArtifactStore`, `train_one_run`, `load_best_policy`, `is_training_complete`.
- Persists: best checkpoint, epoch CSV, and completion manifest atomically.

- [ ] **Step 1: Write failing round-trip and deduplication tests**

```python
import pandas as pd
import torch
from latent_mixup_bc.models import BCPolicy
from latent_mixup_bc.persistence import ArtifactStore


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
    rows = pd.DataFrame([{"dataset_id":"d", "method":"m", "seed":0,
                          "noise_std":0.0, "episode":0, "raw_return":1.0}])
    store.upsert_episode_rows(rows)
    store.upsert_episode_rows(rows.assign(raw_return=2.0))
    saved = pd.read_csv(store.episode_results_path)
    assert len(saved) == 1
    assert saved.raw_return.iloc[0] == 2.0
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_train_persistence.py -q`

Expected: import failure for missing persistence module.

- [ ] **Step 3: Implement atomic artifact writes and all four training branches**

Use temporary files in the destination directory plus `os.replace`. Save model/optimizer/config/normalizer/dimensions/best epoch. Mark a manifest complete only after checkpoint and history exist. Use MSE for all methods, evaluate validation data without augmentation, early-stop on validation MSE, and restore the best checkpoint. A completed run is skipped only if its manifest hash matches and referenced files exist.

- [ ] **Step 4: Verify GREEN and a one-update test**

Run: `pytest tests/test_train_persistence.py -q`

Expected: all tests pass and no non-finite losses.

### Task 5: Matching simulator evaluation and OOD rollouts

**Files:**
- Create: `latent_mixup_bc/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Produces: `reset_env`, `step_env`, `validate_environment`, `evaluate_policy`, `normalized_score`.
- Consumes: restored policy, training normalizer, D4RL-compatible environment, evaluation seeds/noise levels.

- [ ] **Step 1: Write failing API compatibility and noise-location tests**

```python
import numpy as np
from latent_mixup_bc.evaluate import reset_env, step_env


class FiveTupleEnv:
    def reset(self, seed=None): return np.zeros(3), {"seed": seed}
    def step(self, action): return np.ones(3), 2.0, False, True, {}


def test_five_tuple_api_combines_terminated_and_truncated():
    env = FiveTupleEnv()
    obs = reset_env(env, seed=4)
    nxt, reward, done, info = step_env(env, np.zeros(1))
    assert obs.shape == (3,)
    assert reward == 2.0
    assert done is True
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_evaluate.py -q`

Expected: import failure for missing evaluation module.

- [ ] **Step 3: Implement simulator guardrails and rollout**

Support legacy and modern reset/step tuple shapes, but require an explicitly compatible environment ID and D4RL reference score before standard reporting. Add seeded Gaussian noise after converting raw observations with the saved training normalizer, then pass normalized noisy states to the policy. Clip actions, cap episodes at the environment horizon, retain one row per episode, and resume only missing episode keys.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_evaluate.py -q`

Expected: all adapter tests pass.

### Task 6: Statistical reporting, tracker, and figures

**Files:**
- Create: `latent_mixup_bc/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Produces: `summarize_results`, `paired_bootstrap`, `build_run_tracker`, `render_all_figures`.

- [ ] **Step 1: Write failing tests for missing data and paired comparison**

```python
import pandas as pd
from latent_mixup_bc.report import build_run_tracker, summarize_results


def test_missing_run_stays_missing_not_zero():
    expected = pd.DataFrame([
        {"dataset_id":"d", "method":"a", "seed":0},
        {"dataset_id":"d", "method":"b", "seed":0},
    ])
    episodes = pd.DataFrame([
        {"dataset_id":"d", "method":"a", "seed":0, "noise_std":0.0,
         "episode":0, "normalized_score":10.0}
    ])
    tracker = build_run_tracker(expected, episodes, manifests={})
    missing = tracker.loc[tracker.method.eq("b")].iloc[0]
    assert missing.status == "missing"
    assert pd.isna(missing.get("mean_score"))


def test_summary_reports_sample_count():
    df = pd.DataFrame({"dataset_id":["d","d"], "method":["a","a"],
                       "seed":[0,1], "noise_std":[0.0,0.0],
                       "normalized_score":[1.0,3.0]})
    out = summarize_results(df)
    assert out.seed_count.iloc[0] == 2
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_report.py -q`

Expected: import failure for missing reporting module.

- [ ] **Step 3: Implement aggregation without invented values**

Aggregate episode returns to seed means before cross-seed statistics. Bootstrap paired seed differences with a fixed reporting RNG. Render clean grouped bars, robustness curves, relative drop-off, and validation curves in PNG and PDF. Generate the tracker from the 180 expected run keys plus manifests/results and preserve missing values as NaN.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_report.py -q`

Expected: reporting tests pass and figures are non-empty files.

### Task 7: Management Markdown documents

**Files:**
- Create: `README.md`
- Create: `docs/EXPERIMENT_PROTOCOL.md`
- Create: `docs/KAGGLE_RUNBOOK.md`
- Create: `docs/RUN_TRACKER.md`
- Create: `docs/RESULTS_TEMPLATE.md`
- Test: `tests/test_docs.py`

**Interfaces:**
- Consumes: frozen configuration and artifact layout.
- Produces: user-facing research and Kaggle operations documentation.

- [ ] **Step 1: Write failing documentation contract test**

```python
from pathlib import Path


def test_required_management_docs_exist_without_fake_results():
    paths = [Path("README.md"), Path("docs/EXPERIMENT_PROTOCOL.md"),
             Path("docs/KAGGLE_RUNBOOK.md"), Path("docs/RUN_TRACKER.md"),
             Path("docs/RESULTS_TEMPLATE.md")]
    for path in paths:
        assert path.exists()
        assert path.read_text().strip()
    assert "Measured result" in Path("docs/RESULTS_TEMPLATE.md").read_text()
    assert "missing" in Path("docs/RUN_TRACKER.md").read_text().lower()
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_docs.py -q`

Expected: failure because management documents are absent.

- [ ] **Step 3: Write the five documents from the approved design**

Include copy-paste Kaggle resume instructions, the fixed protocol, generated-tracker explanation, explicit Protocol/Measured result/Interpretation labels, and no numerical claims beyond configured run counts.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_docs.py -q`

Expected: documentation contract passes.

### Task 8: Build the standalone Kaggle notebook

**Files:**
- Create: `scripts/build_kaggle_notebook.py`
- Replace: `Latent State-Action Mixup.ipynb`
- Test: `tests/test_notebook.py`

**Interfaces:**
- Consumes: all implementation modules and management-document summaries.
- Produces: one self-contained notebook requiring no local package files at Kaggle runtime.

- [ ] **Step 1: Write failing structural notebook tests**

```python
import json
from pathlib import Path


def test_notebook_has_management_markdown_and_no_legacy_topics():
    nb = json.loads(Path("Latent State-Action Mixup.ipynb").read_text())
    text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    for heading in ["Experiment Dashboard", "Configuration", "D4RL Data",
                    "Models and Augmentation", "Training and Resume",
                    "Evaluation", "Reporting"]:
        assert heading in text
    for forbidden in ["MNIST", "DeepSMOTE", "Convolutional Autoencoder",
                      "make_synthetic_dataset", "KAGGLE_USERNAME", "KAGGLE_KEY"]:
        assert forbidden not in text


def test_notebook_embeds_all_four_methods_and_180_run_assertion():
    text = Path("Latent State-Action Mixup.ipynb").read_text()
    for method in ["vanilla_bc", "noise_bc", "input_mixup_bc", "latent_mixup_bc"]:
        assert method in text
    assert "assert len(runs) == 180" in text
```

- [ ] **Step 2: Verify RED against the existing prototype**

Run: `pytest tests/test_notebook.py -q`

Expected: failures for missing dashboard, methods, resume sections, and legacy setup.

- [ ] **Step 3: Implement deterministic notebook assembly**

Create Markdown + code cell pairs for the ten approved sections. Each Markdown cell lists purpose, inputs, outputs, modes, cost, and recovery. Embed module source in dependency order, add lightweight notebook self-tests, an install/diagnostic cell, configuration cell, scheduler cell, and reporting cell. Default to `SMOKE` so accidental Run All does not start 180 jobs. Do not store execution outputs in the committed notebook.

- [ ] **Step 4: Rebuild and verify GREEN**

Run: `python scripts/build_kaggle_notebook.py && pytest tests/test_notebook.py -q`

Expected: notebook regenerated deterministically and structural tests pass.

### Task 9: End-to-end local verification and Kaggle handoff

**Files:**
- Modify only if verification exposes a root cause: files owned by Tasks 1–8.

**Interfaces:**
- Produces: verified local test report and documented Kaggle-only verification boundary.

- [ ] **Step 1: Run the full local suite**

Run: `pytest -q`

Expected: all tests pass with zero failures.

- [ ] **Step 2: Verify notebook syntax and deterministic generation**

Run: `python scripts/build_kaggle_notebook.py && sha256sum 'Latent State-Action Mixup.ipynb' && python scripts/build_kaggle_notebook.py && sha256sum 'Latent State-Action Mixup.ipynb'`

Expected: both hashes are identical.

- [ ] **Step 3: Run offline notebook cells that do not require network/MuJoCo**

Execute the configuration, self-test, model, persistence, and reporting cells through a temporary notebook copy. Expected: no exception; do not claim D4RL simulator success locally when the compatible backend is absent.

- [ ] **Step 4: Perform Kaggle `SMOKE` acceptance run**

With Internet enabled and the documented compatible D4RL/MuJoCo image/backend, run all cells. Expected: real D4RL data loads; four two-epoch checkpoints restore; clean/noisy episodes complete; CSV/manifests and all figure types are produced. This is an external acceptance step and must remain explicitly unverified until its output is supplied.

- [ ] **Step 5: Refresh management artifacts**

Regenerate `docs/RUN_TRACKER.md` from manifests/results and copy measured smoke-test facts—never placeholder scores—into the measured-results section.

## Plan Self-Review

- Spec coverage: research scope, all four methods, 180 runs, trajectory split, matching simulator, clean/OOD evaluation, resume, reporting, notebook Markdown, and external documents each map to a task.
- Placeholder scan: implementation steps specify concrete interfaces, commands, expectations, and configured values; no benchmark result is pre-filled.
- Type consistency: `ExperimentConfig` and `RunSpec` feed data/training/scheduler; `BCPolicy` and `StateNormalizer` are saved and restored by `ArtifactStore`; episode rows feed reporting and tracker generation.
- Verification boundary: unit/integration checks can run locally; standard D4RL online scores require the matching Kaggle simulator backend and may not be claimed before that run.
