# Kaggle Runbook

## First session

1. Create a Kaggle Notebook with Internet enabled. A GPU is recommended for training; MuJoCo rollouts may remain CPU-bound.
2. Upload `Latent State-Action Mixup.ipynb`.
3. Keep `RUN_MODE = "SMOKE"` and run all cells. This uses one real D4RL dataset, all four methods, two epochs, and two episodes per noise level.
4. Do not switch to `FULL_BENCHMARK` until dependency diagnostics, real-data validation, checkpoint restoration, simulator validation, and plots all succeed.

If Cell 1 reports `numpy.dtype size changed`, the container was polluted by an earlier NumPy downgrade. Use Kaggle **Factory reset session**, upload/open the current notebook, and run from Cell 1. The notebook intentionally does not force-reinstall NumPy or pandas in a clean Kaggle image.

The install cell uses Python 3.12-compatible Gymnasium and native MuJoCo. Training data is downloaded directly from the original D4RL v2 HDF5 URLs. Evaluation uses Gymnasium MuJoCo v5 and is labeled `modern_simulator_compatibility_score`; it is not the original D4RL v2 benchmark.

Prefer a **Tesla T4** accelerator. Current Kaggle PyTorch wheels may not contain kernels for the Tesla P100 (`sm_60`). The trainer checks the actual CUDA architecture and safely falls back to CPU instead of crashing; CPU is acceptable for `SMOKE` but impractical for the full 180-run benchmark.

## Run modes

- `SMOKE`: safe end-to-end acceptance run.
- `SINGLE_RUN`: set dataset, method, and seed to diagnose or fill one missing job.
- `FULL_BENCHMARK`: enumerates 180 training jobs and resumes completed work.

Artifacts default to `/kaggle/working/latent_mixup_bc`. Use `RESUME_INPUT_ROOT = None` in the first session.

## Resume across sessions

1. After a session, use **Save Version** and publish `/kaggle/working/latent_mixup_bc` as a **Kaggle Dataset**.
2. In the next notebook session, attach that Kaggle Dataset as input.
3. Set `RESUME_INPUT_ROOT` to its read-only path under `/kaggle/input/.../latent_mixup_bc`.
4. Run the restore cell. It copies prior checkpoints, histories, manifests, and results into the writable output root without overwriting newer artifacts.
5. Start `FULL_BENCHMARK` or a targeted `SINGLE_RUN`. Completion is accepted only when config hash, checkpoint, history, and manifest agree. Episode CSV uses a full key and resumes missing episodes.

## Recovery

- Dependency failure: preserve logs and stop before training.
- Dataset failure: verify attached HDF5/cache; never use synthetic fallback.
- Interrupted training: rerun; incomplete manifest causes that job to train again.
- Interrupted evaluation: rerun; completed episode keys are deduplicated.
- Config change: a new config hash prevents stale checkpoints from being treated as complete.
