# Latent-Space State–Action Mixup for Behavior Cloning

This repository evaluates latent-space mixup as a data augmentation/regularization method for deterministic Behavior Cloning on D4RL MuJoCo locomotion. It does **not** claim a new offline-RL algorithm.

## Research question

Does latent state–action mixup improve robustness to observation noise without materially sacrificing clean normalized return, relative to Vanilla BC, Gaussian-noise BC, and input-space mixup BC?

## Start here

1. Read the frozen [experiment protocol](docs/EXPERIMENT_PROTOCOL.md).
2. Follow the [Kaggle runbook](docs/KAGGLE_RUNBOOK.md).
3. Open `Latent State-Action Mixup.ipynb` and leave `RUN_MODE="SMOKE"` for the first run.
4. Inspect [run status](docs/RUN_TRACKER.md) and fill only generated measurements into the [results template](docs/RESULTS_TEMPLATE.md).

## Repository map

- `latent_mixup_bc/`: tested configuration, data, models, training, persistence, evaluation, and reporting.
- `tests/`: local regression tests, including mixup pairing and missing-run handling.
- `scripts/build_kaggle_notebook.py`: builds the standalone notebook deterministically.
- `docs/`: protocol, operational runbook, tracker, results template, design, and implementation plan.

## Local verification

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python scripts/build_kaggle_notebook.py
```

The notebook trains on original D4RL v2 HDF5 data and evaluates in modern Gymnasium MuJoCo v5 for Python 3.12 compatibility. Its score is explicitly non-standard and must not be reported as the original D4RL v2 benchmark. Local unit tests do not constitute a Kaggle simulator acceptance run.
