#!/usr/bin/env python3
"""Build the deterministic, standalone Kaggle benchmark notebook."""

from pathlib import Path
from hashlib import sha256

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Latent State-Action Mixup.ipynb"


def management(title, purpose, inputs, outputs, modes, cost, recovery):
    return nbf.v4.new_markdown_cell(
        f"## {title}\n\n"
        f"- **Purpose:** {purpose}\n"
        f"- **Inputs:** {inputs}\n"
        f"- **Outputs:** {outputs}\n"
        f"- **Modes:** {modes}\n"
        f"- **Cost:** {cost}\n"
        f"- **Recovery:** {recovery}\n"
    )


def code(section, source):
    return nbf.v4.new_code_cell(f"# SECTION: {section}\n{source.rstrip()}\n")


def writefile_cell(section, relative_path):
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    return nbf.v4.new_code_cell(f"%%writefile {relative_path}\n# SECTION: {section}\n{source.rstrip()}\n")


def build(output=OUTPUT, include_local_knn=False):
    nb = nbf.v4.new_notebook()
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "kaggle": {"accelerator": "gpu", "internet": True},
    })
    cells = [nbf.v4.new_markdown_cell(
        "# Experiment Dashboard: Latent-Space State–Action Mixup BC\n\n"
        "**Protocol:** Compare five controlled Behavior Cloning augmentations on nine D4RL locomotion datasets. "
        "Evaluation uses a modern Gymnasium v5 simulator and is explicitly labeled a compatibility score, not the original D4RL v2 benchmark.\n\n"
        "| Item | Default |\n|---|---|\n| Run mode | `SMOKE` |\n| Methods | Vanilla, training noise, input mixup, latent mixup |\n"
        "| Full matrix | 9 datasets × 5 methods × 5 seeds = 225 runs |\n| OOD noise | 0, .01, .05, .10, .20 normalized units |\n"
        "| Artifact root | `/kaggle/working/latent_mixup_bc` |\n\n"
        "Run top-to-bottom. First complete `SMOKE`; publish the artifact directory as a Kaggle Dataset, attach it in the next session, set `RESUME_INPUT_ROOT`, then use `FULL_BENCHMARK`."
    )]

    cells += [
        management("Dependency Diagnostics", "Install and verify the Python 3.12-compatible modern backend.", "Kaggle Internet and Python image.", "Imported PyTorch, Gymnasium, native MuJoCo, and data libraries.", "All", "minutes", "Restart the session after installation if imports resolve to stale pre-install packages."),
        code("dependencies", r'''import importlib.util, subprocess, sys

required = {
    "torch": "torch",
    "numpy": "numpy",
    "pandas": "pandas",
    "h5py": "h5py",
    "matplotlib": "matplotlib",
    "gymnasium": "gymnasium[mujoco]>=1.1",
    "mujoco": "mujoco>=3.2",
}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import torch, numpy as np, pandas as pd
import gymnasium as gym, mujoco
gpu = None
if torch.cuda.is_available():
    gpu = {"name": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0), "torch_arches": torch.cuda.get_arch_list()}
print({"python": sys.version, "torch": torch.__version__, "cuda": torch.cuda.is_available(), "gpu": gpu, "gymnasium": gym.__version__, "mujoco": mujoco.__version__})
print("For GPU training, choose a Kaggle T4 accelerator. A P100 may fall back to CPU with current PyTorch wheels.")'''),
        management("Configuration", "Create the frozen benchmark matrix and safe run mode.", "User-selected mode/single-run fields.", "ExperimentConfig, run list, config hash, output paths.", "All", "seconds", "Correct configuration fields and rerun; stale hashes are not resumed."),
        code("configuration package", "from pathlib import Path\nPath('latent_mixup_bc').mkdir(exist_ok=True)\nPath('latent_mixup_bc/__init__.py').write_text('')"),
        writefile_cell("configuration package", "latent_mixup_bc/config.py"),
        code("configuration", r'''from latent_mixup_bc.config import ExperimentConfig, enumerate_runs, config_hash

RUN_MODE = "SMOKE"  # SMOKE | SINGLE_RUN | FULL_BENCHMARK
SINGLE_DATASET = "hopper-medium-v2"
SINGLE_METHOD = "latent_mixup_bc"
SINGLE_SEED = 0
RESUME_INPUT_ROOT = None  # e.g. /kaggle/input/previous-run/latent_mixup_bc
OUTPUT_ROOT = Path("/kaggle/working/latent_mixup_bc") if Path("/kaggle").exists() else Path("artifacts/latent_mixup_bc")

config = ExperimentConfig(
    mode=RUN_MODE,
    single_dataset=SINGLE_DATASET if RUN_MODE == "SINGLE_RUN" else None,
    single_method=SINGLE_METHOD if RUN_MODE == "SINGLE_RUN" else None,
    single_seed=SINGLE_SEED if RUN_MODE == "SINGLE_RUN" else None,
)
runs = enumerate_runs(config)
if RUN_MODE == "FULL_BENCHMARK":
    assert len(runs) == 225
print({"mode": RUN_MODE, "jobs": len(runs), "config_hash": config_hash(config), "output": str(OUTPUT_ROOT)})'''),
        management("Reproducibility and Self-Tests", "Materialize tested core modules and catch semantic regressions before costly work.", "Canonical embedded module source.", "Importable package and passing assertions.", "All", "seconds", "Fix the failing assertion; do not proceed to data or training."),
        writefile_cell("data module", "latent_mixup_bc/data.py"),
        writefile_cell("benchmark compatibility module", "latent_mixup_bc/benchmark.py"),
        writefile_cell("Models and Augmentation", "latent_mixup_bc/models.py"),
        code("self tests", r'''import torch
from latent_mixup_bc.models import BCPolicy, prepare_training_batch

s = torch.tensor([[0.0], [10.0]])
a = torch.tensor([[1.0], [3.0]])
lam, perm = torch.tensor([.25, .75]), torch.tensor([1, 0])
mixed = prepare_training_batch("input_mixup_bc", s, a, lam=lam, permutation=perm)
assert torch.equal(mixed.permutation, perm)
assert torch.allclose(mixed.target, lam[:, None] * a + (1-lam[:, None]) * a[perm])
assert BCPolicy(11, 3)(torch.zeros(2, 11)).shape == (2, 3)
assert set(config.methods) == {"vanilla_bc", "noise_bc", "input_mixup_bc", "latent_mixup_bc", "local_latent_mixup_bc"}
print("Core self-tests passed")'''),
        management("D4RL Data", "Download original fixed D4RL v2 HDF5 data directly, reconstruct trajectories, and compute train-only statistics.", "Kaggle Internet or cached HDF5.", "Validated arrays, disjoint DataLoaders, StateNormalizer.", "All", "minutes", "On download/data failure, attach the cached HDF5 files or repair Internet access; no generated fallback data."),
        code("D4RL Data", r'''from urllib.request import urlretrieve
from latent_mixup_bc.benchmark import dataset_url
from latent_mixup_bc.data import load_d4rl_hdf5, make_loaders

def load_real_d4rl(dataset_id):
    cache_dir = OUTPUT_ROOT / "cache" / "d4rl_hdf5"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{dataset_id}.hdf5"
    if not path.exists():
        temporary = path.with_suffix(".download")
        urlretrieve(dataset_url(dataset_id), temporary)
        temporary.replace(path)
    return load_d4rl_hdf5(path)

dataset_cache = {}
def prepared_data(run):
    if run.dataset_id not in dataset_cache:
        dataset_cache[run.dataset_id] = load_real_d4rl(run.dataset_id)
    cap = config.smoke_max_transitions if config.mode == "SMOKE" else None
    return make_loaders(dataset_cache[run.dataset_id], config.batch_size, config.validation_fraction, run.seed, cap)

sample = dataset_cache.setdefault(runs[0].dataset_id, load_real_d4rl(runs[0].dataset_id))
print(runs[0].dataset_id, sample["observations"].shape, sample["actions"].shape)'''),
        management("Training and Resume", "Train fair ablations with early stopping and durable artifacts.", "Prepared D4RL loaders, config, optional prior artifacts.", "Checkpoints, histories, completion manifests.", "All", "expensive", "Interrupted/incomplete jobs rerun; complete matching-hash jobs are skipped."),
        writefile_cell("persistence module", "latent_mixup_bc/persistence.py"),
        writefile_cell("training module", "latent_mixup_bc/train.py"),
        code("Training and Resume", r'''from latent_mixup_bc.persistence import ArtifactStore, restore_missing_artifacts
from latent_mixup_bc.train import train_one_run

if RESUME_INPUT_ROOT:
    source = Path(RESUME_INPUT_ROOT)
    if not source.exists():
        raise FileNotFoundError(f"RESUME_INPUT_ROOT does not exist: {source}")
    copied = restore_missing_artifacts(source, OUTPUT_ROOT)
    print(f"Restored {len(copied)} missing artifacts without overwriting working files")

store = ArtifactStore(OUTPUT_ROOT)
for run in runs:
    train_loader, valid_loader, normalizer, train_rows, valid_rows = prepared_data(run)
    assert set(train_rows).isdisjoint(valid_rows)
    state_dim = train_loader.dataset.tensors[0].shape[1]
    action_dim = train_loader.dataset.tensors[1].shape[1]
    result = train_one_run(run, config, train_loader, valid_loader, state_dim, action_dim, store, normalizer=normalizer)
    print(run, result)'''),
        management("Evaluation", "Restore each best policy and collect clean/OOD returns in the modern v5 compatibility simulator.", "Gymnasium MuJoCo v5, checkpoints, saved normalizer.", "Episode CSV with raw return and explicitly labeled compatibility score.", "All", "expensive", "Fix any version/dimension failure; never relabel this result as the original D4RL v2 benchmark."),
        writefile_cell("evaluation module", "latent_mixup_bc/evaluate.py"),
        code("Evaluation", r'''from latent_mixup_bc.data import StateNormalizer
from latent_mixup_bc.evaluate import evaluate_policy
from latent_mixup_bc.train import load_best_policy

existing = pd.read_csv(store.episode_results_path) if store.episode_results_path.exists() else pd.DataFrame()
episodes = 2 if config.mode == "SMOKE" else config.evaluation_episodes
for run in runs:
    model, payload = load_best_policy(run, config, store)
    normalizer = StateNormalizer.from_dict(payload["normalizer"])
    env = gym.make(run.env_id)
    for noise_std in config.noise_levels:
        completed = set()
        if not existing.empty:
            match = existing.dataset_id.eq(run.dataset_id) & existing.method.eq(run.method) & existing.seed.eq(run.seed) & existing.noise_std.eq(noise_std)
            completed = set(existing.loc[match, "episode"].astype(int))
        rows = evaluate_policy(env, model, normalizer, run.dataset_id, run.method, run.seed, noise_std, episodes, run.env_id)
        rows = [row for row in rows if row["episode"] not in completed]
        if rows:
            store.upsert_episode_rows(pd.DataFrame(rows))
            existing = pd.read_csv(store.episode_results_path)
    env.close()
print("Episode rows:", len(pd.read_csv(store.episode_results_path)))'''),
        management("Benchmark Scheduler", "Show run coverage and target missing work without confusing absence with zero.", "Expected run matrix, manifests, episode CSV.", "Status table for missing/trained/partial/complete jobs.", "All", "seconds", "Use SINGLE_RUN for failed/missing rows or continue FULL_BENCHMARK."),
        code("scheduler", r'''import json
expected = pd.DataFrame([vars(run) for run in runs])
manifests = {}
for run in runs:
    path = store.manifest_path(run.dataset_id, run.method, run.seed)
    if path.exists():
        manifests[(run.dataset_id, run.method, run.seed)] = json.loads(path.read_text())
print("Training manifests:", len(manifests), "/", len(runs))'''),
        management("Reporting", "Aggregate at seed level, quantify uncertainty, and render clean/OOD comparisons.", "Episode CSV, histories, expected matrix.", "summary.csv, run tracker, PNG/PDF figures.", "All", "minutes", "Incomplete runs remain NaN. Add runs rather than imputing missing scores."),
        writefile_cell("reporting module", "latent_mixup_bc/report.py"),
        code("Reporting", r'''from latent_mixup_bc.report import build_run_tracker, render_all_figures, summarize_results

episode_results = pd.read_csv(store.episode_results_path)
summary = summarize_results(episode_results)
store.write_csv_atomic(summary, OUTPUT_ROOT / "results" / "summary.csv")
tracker = build_run_tracker(expected[["dataset_id", "method", "seed"]], episode_results, manifests)
store.write_csv_atomic(tracker, OUTPUT_ROOT / "results" / "run_tracker.csv")

history_frames = []
for run in runs:
    path = store.history_path(run.dataset_id, run.method, run.seed)
    if path.exists():
        frame = pd.read_csv(path)
        frame["dataset_id"], frame["method"], frame["seed"] = run.dataset_id, run.method, run.seed
        history_frames.append(frame)
histories = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
figure_paths = render_all_figures(episode_results, histories, OUTPUT_ROOT / "figures")
display(summary)
display(tracker.status.value_counts(dropna=False))
print("Figures:", [str(path) for path in figure_paths])'''),
        management("Display Saved Result Figures", "Show the generated result plots inline in Kaggle while retaining PNG/PDF files.", "PNG files created by the Reporting cell.", "Four inline figures: clean scores, robustness, drop-off, and validation loss.", "All", "seconds", "Run the Reporting cell first if any PNG is missing."),
        code("Display Saved Result Figures", r'''from IPython.display import Image, display
from IPython.display import Markdown

INLINE_FIGURES = [
    ("Clean Scores", "clean_scores.png"),
    ("Robustness to Observation Noise", "robustness.png"),
    ("Relative Performance Drop-off", "dropoff.png"),
    ("Validation Loss", "validation_loss.png"),
]
for title, filename in INLINE_FIGURES:
    path = OUTPUT_ROOT / "figures" / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the Reporting cell first")
    display(Markdown(f"### {title}"))
    display(Image(filename=str(path)))'''),
        management("Incremental Pilot Comparison", "Define a resumable helper that trains and evaluates one additional method without rerunning completed methods or earlier notebook setup.", "Existing SINGLE_RUN config, prepared dataset cache, ArtifactStore, and Vanilla BC artifacts.", "Reusable run_pilot_method(method) helper.", "SINGLE_RUN only", "seconds", "Run the notebook through the first Reporting section once; then these pilot cells can be rerun independently."),
        code("Incremental Pilot Comparison", r'''from dataclasses import replace

if RUN_MODE != "SINGLE_RUN":
    raise RuntimeError("Pilot requires RUN_MODE='SINGLE_RUN'")

def run_pilot_method(method):
    if method not in {"noise_bc", "input_mixup_bc", "latent_mixup_bc"}:
        raise ValueError(f"Unsupported incremental pilot method: {method}")
    pilot_config = replace(config, single_method=method)
    pilot_run = enumerate_runs(pilot_config)[0]
    train_loader, valid_loader, normalizer, train_rows, valid_rows = prepared_data(pilot_run)
    assert set(train_rows).isdisjoint(valid_rows)
    state_dim = train_loader.dataset.tensors[0].shape[1]
    action_dim = train_loader.dataset.tensors[1].shape[1]
    train_result = train_one_run(
        pilot_run, pilot_config, train_loader, valid_loader,
        state_dim, action_dim, store, normalizer=normalizer,
    )
    print(f"{method} training:", train_result)

    model, payload = load_best_policy(pilot_run, pilot_config, store)
    saved_normalizer = StateNormalizer.from_dict(payload["normalizer"])
    env = gym.make(pilot_run.env_id)
    required_episodes = pilot_config.evaluation_episodes
    for noise_std in pilot_config.noise_levels:
        current = pd.read_csv(store.episode_results_path) if store.episode_results_path.exists() else pd.DataFrame()
        completed = set()
        if not current.empty:
            match = (
                current.dataset_id.eq(pilot_run.dataset_id)
                & current.method.eq(method)
                & current.seed.eq(pilot_run.seed)
                & current.noise_std.eq(noise_std)
            )
            completed = set(current.loc[match, "episode"].astype(int))
        missing = set(range(required_episodes)) - completed
        if not missing:
            print(f"{method} noise={noise_std}: evaluation already complete")
            continue
        rows = evaluate_policy(
            env, model, saved_normalizer, pilot_run.dataset_id, method,
            pilot_run.seed, noise_std, required_episodes, pilot_run.env_id,
        )
        store.upsert_episode_rows(pd.DataFrame([row for row in rows if row["episode"] in missing]))
        print(f"{method} noise={noise_std}: added {len(missing)} episodes")
    env.close()
    return train_result'''),
        management("Pilot Method: Noise BC", "Train/evaluate Gaussian-noise BC for the selected dataset and seed.", "Incremental pilot helper and existing artifacts.", "Noise BC checkpoint, history, manifest, and episode rows.", "SINGLE_RUN only", "expensive", "Rerun safely after interruption; completed training and evaluation keys are skipped."),
        code("Pilot Method: Noise BC", 'noise_result = run_pilot_method("noise_bc")'),
        management("Pilot Method: Input Mixup BC", "Train/evaluate input-space state-action mixup BC for the selected dataset and seed.", "Incremental pilot helper and existing artifacts.", "Input Mixup checkpoint, history, manifest, and episode rows.", "SINGLE_RUN only", "expensive", "Rerun safely after interruption; completed training and evaluation keys are skipped."),
        code("Pilot Method: Input Mixup BC", 'input_mixup_result = run_pilot_method("input_mixup_bc")'),
        management("Pilot Method: Latent Mixup BC", "Train/evaluate the proposed latent-space state-action mixup BC for the selected dataset and seed.", "Incremental pilot helper and existing artifacts.", "Latent Mixup checkpoint, history, manifest, and episode rows.", "SINGLE_RUN only", "expensive", "Rerun safely after interruption; completed training and evaluation keys are skipped."),
        code("Pilot Method: Latent Mixup BC", 'latent_mixup_result = run_pilot_method("latent_mixup_bc")'),
        management("Pilot Comparison Plots", "Regenerate and display the four-method comparison for the selected dataset and seed.", "Episode CSV and four method history CSV files.", "Updated summary plus inline clean, robustness, drop-off, and validation-loss figures.", "SINGLE_RUN only", "seconds", "Run any missing pilot method cell first; absent methods are left missing rather than treated as zero."),
        code("Pilot Comparison Plots", r'''PILOT_METHODS = ["vanilla_bc", "noise_bc", "input_mixup_bc", "latent_mixup_bc"]
all_episode_results = pd.read_csv(store.episode_results_path)
pilot_episode_results = all_episode_results.loc[
    all_episode_results.dataset_id.eq(SINGLE_DATASET)
    & all_episode_results.seed.eq(SINGLE_SEED)
    & all_episode_results.method.isin(PILOT_METHODS)
].copy()
if pilot_episode_results.empty:
    raise RuntimeError("No pilot episode results found")

pilot_summary = summarize_results(pilot_episode_results)
display(pilot_summary)
available_methods = set(pilot_episode_results.method.unique())
missing_methods = set(PILOT_METHODS) - available_methods
if missing_methods:
    print("Methods still missing from comparison:", sorted(missing_methods))

pilot_histories = []
for method in PILOT_METHODS:
    path = store.history_path(SINGLE_DATASET, method, SINGLE_SEED)
    if path.exists():
        frame = pd.read_csv(path)
        frame["method"] = method
        pilot_histories.append(frame)
pilot_histories = pd.concat(pilot_histories, ignore_index=True) if pilot_histories else pd.DataFrame()

pilot_figure_paths = render_all_figures(
    pilot_episode_results, pilot_histories, OUTPUT_ROOT / "figures" / "pilot_comparison"
)
for title, filename in INLINE_FIGURES:
    path = OUTPUT_ROOT / "figures" / "pilot_comparison" / filename
    if path.exists():
        display(Markdown(f"### Pilot Comparison — {title}"))
        display(Image(filename=str(path)))'''),
        management("Multi-Seed Pilot", "Define resumable training/evaluation helpers for seeds 1–4 across all four methods.", "Existing dataset cache, ArtifactStore, model modules, and completed seed 0 artifacts.", "run_method_seed and run_seed helpers.", "SINGLE_RUN only", "seconds", "Run one seed cell per Kaggle session if needed; every method/noise/episode resumes independently."),
        code("Multi-Seed Pilot", r'''SEEDS = [0, 1, 2, 3, 4]
MULTI_SEED_METHODS = ["vanilla_bc", "noise_bc", "input_mixup_bc", "latent_mixup_bc"]

if RUN_MODE != "SINGLE_RUN":
    raise RuntimeError("Multi-seed pilot requires RUN_MODE='SINGLE_RUN'")

def run_method_seed(method, seed):
    if method not in MULTI_SEED_METHODS:
        raise ValueError(f"Unknown method: {method}")
    if seed not in SEEDS:
        raise ValueError(f"Seed must be one of {SEEDS}")
    method_config = replace(config, single_method=method, single_seed=seed)
    method_run = enumerate_runs(method_config)[0]
    train_loader, valid_loader, normalizer, train_rows, valid_rows = prepared_data(method_run)
    assert set(train_rows).isdisjoint(valid_rows)
    state_dim = train_loader.dataset.tensors[0].shape[1]
    action_dim = train_loader.dataset.tensors[1].shape[1]
    train_result = train_one_run(
        method_run, method_config, train_loader, valid_loader,
        state_dim, action_dim, store, normalizer=normalizer,
    )
    print(f"seed={seed} method={method} training:", train_result)

    model, payload = load_best_policy(method_run, method_config, store)
    saved_normalizer = StateNormalizer.from_dict(payload["normalizer"])
    env = gym.make(method_run.env_id)
    required_episodes = method_config.evaluation_episodes
    for noise_std in method_config.noise_levels:
        current = pd.read_csv(store.episode_results_path) if store.episode_results_path.exists() else pd.DataFrame()
        completed = set()
        if not current.empty:
            match = (
                current.dataset_id.eq(method_run.dataset_id)
                & current.method.eq(method)
                & current.seed.eq(seed)
                & current.noise_std.eq(noise_std)
            )
            completed = set(current.loc[match, "episode"].astype(int))
        missing = set(range(required_episodes)) - completed
        if not missing:
            print(f"seed={seed} method={method} noise={noise_std}: complete")
            continue
        rows = evaluate_policy(
            env, model, saved_normalizer, method_run.dataset_id, method,
            seed, noise_std, required_episodes, method_run.env_id,
        )
        new_rows = [row for row in rows if row["episode"] in missing]
        store.upsert_episode_rows(pd.DataFrame(new_rows))
        print(f"seed={seed} method={method} noise={noise_std}: added {len(new_rows)} episodes")
    env.close()
    return train_result

def run_seed(seed):
    results = {}
    for method in MULTI_SEED_METHODS:
        results[method] = run_method_seed(method, seed)
    return results'''),
        management("Pilot Seed 1", "Train and evaluate all four methods for seed 1.", "Multi-seed helper and existing seed 0 artifacts.", "Four checkpoints/manifests/histories and 400 episode rows for seed 1.", "SINGLE_RUN only", "expensive", "Safe to rerun; completed method and episode keys are skipped."),
        code("Pilot Seed 1", "seed_1_results = run_seed(1)"),
        management("Pilot Seed 2", "Train and evaluate all four methods for seed 2.", "Multi-seed helper and prior artifacts.", "Four checkpoints/manifests/histories and 400 episode rows for seed 2.", "SINGLE_RUN only", "expensive", "Safe to rerun; completed method and episode keys are skipped."),
        code("Pilot Seed 2", "seed_2_results = run_seed(2)"),
        management("Pilot Seed 3", "Train and evaluate all four methods for seed 3.", "Multi-seed helper and prior artifacts.", "Four checkpoints/manifests/histories and 400 episode rows for seed 3.", "SINGLE_RUN only", "expensive", "Safe to rerun; completed method and episode keys are skipped."),
        code("Pilot Seed 3", "seed_3_results = run_seed(3)"),
        management("Pilot Seed 4", "Train and evaluate all four methods for seed 4.", "Multi-seed helper and prior artifacts.", "Four checkpoints/manifests/histories and 400 episode rows for seed 4.", "SINGLE_RUN only", "expensive", "Safe to rerun; completed method and episode keys are skipped."),
        code("Pilot Seed 4", "seed_4_results = run_seed(4)"),
        management("Multi-Seed Aggregate Plots", "Aggregate completed seeds without imputing missing runs and display uncertainty-aware comparisons.", "Episode CSV and history CSV files for seeds 0–4.", "Coverage table, mean/std/CI table, and four inline aggregate figures.", "SINGLE_RUN only", "seconds", "This cell can run after any seed; incomplete seeds remain absent and are reported in coverage."),
        code("Multi-Seed Aggregate Plots", r'''all_episode_results = pd.read_csv(store.episode_results_path)
multi_seed_episodes = all_episode_results.loc[
    all_episode_results.dataset_id.eq(SINGLE_DATASET)
    & all_episode_results.seed.isin(SEEDS)
    & all_episode_results.method.isin(MULTI_SEED_METHODS)
].copy()

clean_rows = multi_seed_episodes.loc[multi_seed_episodes.noise_std.eq(0.0)]
completed_seed_count = clean_rows.groupby("method")["seed"].nunique().reindex(MULTI_SEED_METHODS, fill_value=0)
display(Markdown("### Completed training seeds by method"))
display(completed_seed_count.rename("completed_seed_count").to_frame())

multi_seed_summary = summarize_results(multi_seed_episodes)
display(Markdown("### Multi-seed summary"))
display(multi_seed_summary)

multi_seed_histories = []
for seed in SEEDS:
    for method in MULTI_SEED_METHODS:
        path = store.history_path(SINGLE_DATASET, method, seed)
        if path.exists():
            frame = pd.read_csv(path)
            frame["method"] = method
            frame["seed"] = seed
            multi_seed_histories.append(frame)
multi_seed_histories = pd.concat(multi_seed_histories, ignore_index=True) if multi_seed_histories else pd.DataFrame()

aggregate_dir = OUTPUT_ROOT / "figures" / "multi_seed_comparison"
aggregate_paths = render_all_figures(multi_seed_episodes, multi_seed_histories, aggregate_dir)
for title, filename in INLINE_FIGURES:
    path = aggregate_dir / filename
    if path.exists():
        display(Markdown(f"### Multi-Seed — {title}"))
        display(Image(filename=str(path)))'''),
        management("Export and Next Session", "Validate durable outputs before publishing them for resume.", "Artifact directory.", "Integrity summary ready for Kaggle Dataset publication.", "All", "seconds", "Do not publish temporary files; rerun the owning section if a required artifact is absent."),
        code("export", r'''temporary_files = list(OUTPUT_ROOT.rglob("*.tmp"))
assert not temporary_files, temporary_files
print("Artifact root:", OUTPUT_ROOT)
print("Publish this directory as a Kaggle Dataset, attach it next session, then set RESUME_INPUT_ROOT.")'''),
    ]
    if include_local_knn:
        cells[0].source = cells[0].source.replace(
            "Latent-Space State–Action Mixup BC",
            "In-batch k-NN Latent State–Action Mixup BC",
        )
        for cell in cells:
            if cell.cell_type == "code" and 'RUN_MODE = "SMOKE"' in cell.source:
                cell.source = cell.source.replace('RUN_MODE = "SMOKE"', 'RUN_MODE = "SINGLE_RUN"')
                break
        cells[0].source += (
            "\n\n> **Extension workflow:** attach/preserve the artifact directory from the completed "
            "four-method experiment. This notebook skips old training/evaluation and trains only "
            "`local_latent_mixup_bc` for seeds 0–4."
        )
        for cell in cells:
            if cell.cell_type != "code":
                continue
            if cell.source.startswith("# SECTION: Training and Resume"):
                cell.source = r'''# SECTION: Training and Resume
from latent_mixup_bc.persistence import ArtifactStore, restore_missing_artifacts

if RESUME_INPUT_ROOT:
    source = Path(RESUME_INPUT_ROOT)
    if not source.exists():
        raise FileNotFoundError(f"RESUME_INPUT_ROOT does not exist: {source}")
    copied = restore_missing_artifacts(source, OUTPUT_ROOT)
    print(f"Restored {len(copied)} missing artifacts without overwriting working files")
store = ArtifactStore(OUTPUT_ROOT)
print("Extension notebook: baseline training skipped; existing artifacts will be reused.")
'''
            elif cell.source.startswith("# SECTION: Evaluation"):
                cell.source = '''# SECTION: Evaluation\nprint("Extension notebook: baseline evaluation skipped; existing episode CSV will be reused.")\n'''
            elif any(cell.source.startswith(f"# SECTION: Pilot Method: {name}") for name in ("Noise BC", "Input Mixup BC", "Latent Mixup BC")):
                cell.source = '''# SECTION: Old Method Pilot\nprint("Old method pilot skipped; restored artifacts are reused.")\n'''
            elif any(cell.source.startswith(f"# SECTION: Pilot Seed {seed}") for seed in range(1, 5)):
                cell.source = '''# SECTION: Old Multi-Seed Pilot\nprint("Old multi-seed training skipped; restored artifacts are reused.")\n'''
        local_cells = [
            management("In-batch k-NN Latent State–Action Mixup", "Add the local action-compatible nearest-neighbor method while preserving the four completed baselines.", "Existing four-method artifacts, embedded updated model/training modules, and selected dataset.", "Local k-NN method configuration and incremental seed cells.", "SINGLE_RUN only", "seconds", "Attach/preserve prior artifacts; only the new method is trained in the following seed cells."),
            code("In-batch k-NN Latent State–Action Mixup", r'''LOCAL_KNN_METHOD = "local_latent_mixup_bc"
if LOCAL_KNN_METHOD not in MULTI_SEED_METHODS:
    MULTI_SEED_METHODS.append(LOCAL_KNN_METHOD)
print({
    "method": LOCAL_KNN_METHOD,
    "mixup_alpha": config.mixup_alpha,
    "action_threshold": config.action_threshold,
    "seeds": SEEDS,
})'''),
        ]
        for seed in range(5):
            local_cells.extend([
                management(
                    f"Local k-NN Seed {seed}",
                    f"Train and evaluate only local_latent_mixup_bc for seed {seed}.",
                    "Updated helper, dataset cache, and prior baseline artifacts.",
                    f"Local k-NN checkpoint/history/manifest and evaluation rows for seed {seed}.",
                    "SINGLE_RUN only",
                    "expensive",
                    "Safe to rerun; matching checkpoint and episode keys are skipped.",
                ),
                code(
                    f"Local k-NN Seed {seed}",
                    f'local_knn_seed_{seed}_result = run_method_seed("local_latent_mixup_bc", {seed})',
                ),
            ])
        local_cells.extend([
            management("Five-Method Aggregate Comparison", "Compare local k-NN mixup with Vanilla, Noise, Input Mixup, and random Latent Mixup across all completed seeds.", "Episode and history artifacts for five methods and seeds 0–4.", "Coverage, mean/std/CI table, and inline aggregate figures.", "SINGLE_RUN only", "seconds", "Missing local seeds stay missing; rerun their dedicated cell rather than imputing scores."),
            code("Five-Method Aggregate Comparison", r'''FIVE_METHODS = [
    "vanilla_bc", "noise_bc", "input_mixup_bc",
    "latent_mixup_bc", "local_latent_mixup_bc",
]
all_results = pd.read_csv(store.episode_results_path)
five_method_episodes = all_results.loc[
    all_results.dataset_id.eq(SINGLE_DATASET)
    & all_results.seed.isin(SEEDS)
    & all_results.method.isin(FIVE_METHODS)
].copy()

clean_rows = five_method_episodes.loc[five_method_episodes.noise_std.eq(0.0)]
five_method_coverage = clean_rows.groupby("method")["seed"].nunique().reindex(FIVE_METHODS, fill_value=0)
display(Markdown("### Five-method seed coverage"))
display(five_method_coverage.rename("completed_seed_count").to_frame())

five_method_summary = summarize_results(five_method_episodes)
display(Markdown("### Five-method summary"))
display(five_method_summary)

five_method_histories = []
for seed in SEEDS:
    for method in FIVE_METHODS:
        path = store.history_path(SINGLE_DATASET, method, seed)
        if path.exists():
            frame = pd.read_csv(path)
            frame["method"] = method
            frame["seed"] = seed
            five_method_histories.append(frame)
five_method_histories = pd.concat(five_method_histories, ignore_index=True) if five_method_histories else pd.DataFrame()

comparison_dir = OUTPUT_ROOT / "figures" / "five_method_comparison"
comparison_paths = render_all_figures(
    five_method_episodes, five_method_histories, comparison_dir
)
for title, filename in INLINE_FIGURES:
    path = comparison_dir / filename
    if path.exists():
        display(Markdown(f"### Five Methods — {title}"))
        display(Image(filename=str(path)))'''),
        ])
        cells[-2:-2] = local_cells

    managed_cells = []
    for cell in cells:
        if cell.cell_type == "code" and managed_cells and managed_cells[-1].cell_type != "markdown":
            section_name = cell.source.splitlines()[0].replace("# SECTION:", "").strip()
            managed_cells.append(management(
                f"{section_name.title()} (continued)",
                "Continue the preceding section with one independently rerunnable unit.",
                "Objects and files produced by the preceding cell.",
                "Updated module or runtime state for the next cell.",
                "All",
                "seconds to minutes",
                "Rerun the preceding cell, then rerun this unit; durable artifacts are not deleted.",
            ))
        managed_cells.append(cell)
    for index, cell in enumerate(managed_cells):
        identity = f"{index}\0{cell.cell_type}\0{cell.source}".encode("utf-8")
        cell["id"] = sha256(identity).hexdigest()[:16]
    nb["cells"] = managed_cells
    Path(output).write_text(nbf.writes(nb), encoding="utf-8")


if __name__ == "__main__":
    build()
