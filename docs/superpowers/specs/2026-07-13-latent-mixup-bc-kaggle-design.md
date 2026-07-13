# Latent-Space State–Action Mixup for Behavior Cloning

## 1. Research claim

This project evaluates latent-space state–action mixup as a data augmentation and regularization method for offline Behavior Cloning (BC). It does not claim to introduce a new offline reinforcement-learning algorithm.

The primary hypothesis is:

> Mixing encoded states and their corresponding actions improves policy robustness to observation perturbations without materially reducing clean-environment performance, compared with vanilla BC and simpler augmentation baselines.

Evidence must come from matched ablations across multiple D4RL locomotion datasets and random seeds. A lower validation MSE alone is not sufficient evidence; the main outcome is normalized online return in the matching simulator.

## 2. Experimental scope

### Environments and datasets

Use the D4RL MuJoCo locomotion benchmark with all nine combinations below:

- Environments: `hopper`, `halfcheetah`, `walker2d`.
- Dataset qualities: `medium`, `medium-replay`, `medium-expert`.
- Dataset/environment version: D4RL `v2` wherever the backend exposes it.

The loader must never silently replace missing D4RL data with synthetic data. A missing dataset, incompatible environment, or failed dependency check must stop the affected run with an actionable error.

### Methods

Train five methods:

1. `vanilla_bc`: unmodified state input and action target.
2. `noise_bc`: Gaussian noise added to normalized states during training.
3. `input_mixup_bc`: state vectors and action targets mixed with the same coefficient and permutation.
4. `latent_mixup_bc`: encoded state representations and action targets mixed with the same coefficient and permutation before the policy head.
5. `local_latent_mixup_bc`: encoded states mix only with the nearest in-batch latent neighbor satisfying an action-distance threshold; samples without a valid neighbor remain unchanged.

All methods must use the same encoder/head capacity. Vanilla, noise, and input-mixup methods still pass through the same encoder and policy head; only the augmentation location differs. This controls for parameter count and depth.

### Repetitions and evaluation

- Training seeds: `0, 1, 2, 3, 4`.
- Evaluation episodes: 20 per trained seed and noise level by default.
- Observation-noise levels: `0.00, 0.01, 0.05, 0.10, 0.20` in normalized-state units.
- Primary metric: `modern_simulator_compatibility_score`, reported as mean and standard deviation over seeds. It scales Gymnasium MuJoCo v5 returns with published D4RL references and is not a standard D4RL v2 score.
- Supporting metrics: raw return, validation MSE, relative score drop from clean evaluation, and 95% bootstrap confidence intervals over seed-level means.

The notebook must retain episode-level results so alternative aggregate statistics can be computed without rerunning simulation.

## 3. Fair-comparison protocol

Each `(dataset, seed)` comparison shares:

- trajectory-based train/validation split;
- network architecture and initialization seed;
- shuffled batch ordering where augmentation semantics permit it;
- optimizer, learning rate, batch size, maximum epochs, early-stopping rule, and action clipping;
- state normalization computed only from the training trajectories;
- model-selection criterion based on validation MSE;
- evaluation episode seeds.

Transitions must be grouped into trajectories using `terminals` and `timeouts` before splitting. Random transition-level splitting is prohibited because adjacent transitions would leak across train and validation sets.

Mixup samples `lambda ~ Beta(alpha, alpha)`. One permutation is generated per batch and reused for both the state/latent representation and action target. The implementation must expose the permutation in a testable return value to prevent mismatched-pair regressions.

## 4. Model architecture and losses

The common policy consists of:

- encoder: state dimension → 256 → 256 → latent dimension, with ReLU activations;
- policy head: latent dimension → 256 → 256 → action dimension, with ReLU activations in hidden layers;
- deterministic continuous action output, clipped to the simulator action bounds during evaluation.

Default latent dimension is 128. The loss for every method is mean squared error. For either mixup method, the regression target is

`a_mix = lambda * a + (1 - lambda) * a[permutation]`.

For input mixup, the policy consumes the equivalently mixed normalized state. For latent mixup, the encoder processes original normalized states first, then the policy head consumes the mixed latent representation.

## 5. Kaggle execution architecture

The final artifact remains a single readable Kaggle notebook, organized into sections that can be run top to bottom:

1. Configuration and run mode.
2. Dependency installation and backend diagnostics.
3. Reproducibility utilities and self-tests.
4. D4RL dataset discovery, loading, validation, and caching.
5. Trajectory split and preprocessing.
6. Augmentation functions and policy definitions.
7. Training, early stopping, and checkpoint persistence.
8. Simulator adapter and clean/OOD evaluation.
9. Resumable benchmark scheduler.
10. Aggregation, statistical reporting, and visualizations.

Each section must begin with a Markdown cell that states:

- the purpose of the section;
- inputs and artifacts it consumes;
- outputs and artifacts it produces;
- whether the section is required in `SMOKE`, `SINGLE_RUN`, and `FULL_BENCHMARK` modes;
- expected runtime or cost category (`seconds`, `minutes`, or `expensive`);
- common failure modes and the safe recovery/resume action.

The notebook must also include an opening experiment dashboard in Markdown. It summarizes the research hypothesis, current run mode, benchmark matrix, artifact root, and the order in which a Kaggle user should execute or resume cells. Long implementation details belong in the external documentation rather than duplicating large prose blocks in the notebook.

Three run modes are required:

- `SMOKE`: one dataset, one seed, small subset, two epochs, two evaluation episodes.
- `SINGLE_RUN`: one explicitly selected dataset/method/seed.
- `FULL_BENCHMARK`: all nine datasets, five methods, and five seeds.

The full benchmark is expected to span multiple Kaggle sessions. Each run is identified by `(dataset_id, method, seed, config_hash)`. A run is skipped only when its checkpoint, training history, completion manifest, and required evaluation rows all exist and pass integrity checks.

## 6. Persistence and resume behavior

Use a configurable output root, defaulting to `/kaggle/working/latent_mixup_bc` on Kaggle and `artifacts/latent_mixup_bc` locally. Store:

- `checkpoints/<dataset>/<method>/seed_<n>.pth`;
- `histories/<dataset>/<method>/seed_<n>.csv`;
- `manifests/<dataset>/<method>/seed_<n>.json`;
- `results/episode_results.csv`;
- `results/summary.csv`;
- `figures/*.png` and `figures/*.pdf`;
- cached processed datasets and split indices keyed by dataset and seed.

Writes must be atomic: write a temporary file in the destination directory and rename it after success. A manifest is marked complete only after the best checkpoint and training history are durable. CSV updates must deduplicate by their full experimental key, allowing interrupted evaluation to resume at the missing episode.

Because `/kaggle/working` is ephemeral between sessions, the notebook must include instructions for publishing the output directory as a Kaggle Dataset and attaching it as input to the next session. It must accept a read-only resume input directory and copy only the required prior artifacts into the current writable output directory.

## 7. D4RL and simulator compatibility

Training data access and online simulation must be separate adapters.

- The dataset adapter loads the exact D4RL HDF5 data, validates required keys and shapes, and can cache processed NumPy arrays independently of MuJoCo.
- The simulator adapter targets Gymnasium MuJoCo v5 on Kaggle Python 3.12 and uses published D4RL references only to form an explicitly labeled compatibility score.
- Before evaluation, a compatibility smoke test checks observation dimension, action dimension, finite reset output, action bounds, one successful step, and availability of reference scores.
- The notebook must not present a D4RL `v2` policy evaluated in Gymnasium `v5` as a standard D4RL score.

The supported Kaggle backend is Gymnasium MuJoCo v5. Training datasets remain original D4RL v2 HDF5 files downloaded directly from the official URLs. Exact legacy v2 simulator reproduction is outside this Python 3.12 implementation scope.

## 8. Statistical analysis and figures

Generate at least:

1. Clean normalized-score grouped bar chart with uncertainty bars.
2. Robustness curve: normalized score versus observation-noise level.
3. Relative drop-off chart from clean score for each method.
4. Validation-loss learning curves.
5. Per-dataset and aggregate result tables with mean, standard deviation, confidence interval, and run count.

The principal comparison is latent mixup versus each baseline, paired by dataset and seed. Report paired seed-level differences with bootstrap confidence intervals. Do not describe latent mixup as superior if confidence intervals are inconclusive or if gains occur only after averaging away substantial task-specific regressions.

## 9. Validation and self-tests

Notebook self-tests must run before expensive training and cover:

- deterministic seeding;
- trajectory reconstruction and split disjointness;
- normalization using training statistics only;
- mixup output shapes and endpoint behavior;
- identical permutation and coefficient use for representation and target;
- forward-pass shape for every method;
- finite MSE and one optimizer update;
- checkpoint round trip with identical predictions;
- resume-key and CSV deduplication behavior;
- environment API handling for both four-value and five-value `step` signatures when applicable.

`SMOKE` mode is the end-to-end acceptance test on Kaggle. It must download/load real D4RL data, train all four methods briefly, restore their checkpoints, complete clean and noisy simulator episodes, write results, and render every figure type.

## 10. Research-management documentation

In addition to Markdown cells in the notebook, maintain these repository documents:

- `README.md`: project overview, research claim, repository map, quick start, and links to the documents below.
- `docs/EXPERIMENT_PROTOCOL.md`: frozen datasets, methods, shared hyperparameters, seed policy, clean/OOD evaluation, metrics, statistical tests, and rules for interpreting results.
- `docs/KAGGLE_RUNBOOK.md`: Kaggle accelerator/internet settings, dependency setup, run modes, artifact export, publishing output as a Kaggle Dataset, attaching prior output, and recovery from interrupted sessions.
- `docs/RUN_TRACKER.md`: generated/refreshable 180-run matrix with columns for dataset, method, seed, configuration hash, training status, evaluation coverage, checkpoint, and notes. The notebook remains the source of truth and updates status from manifests/results rather than requiring manual duplication.
- `docs/RESULTS_TEMPLATE.md`: empty reporting structure for aggregate tables, clean/OOD figures, paired comparisons, limitations, and conclusion. It must not include invented numbers.

Documentation must distinguish three kinds of content using explicit labels where relevant:

- **Protocol:** decisions fixed before seeing benchmark results.
- **Measured result:** values loaded from generated result artifacts.
- **Interpretation:** conclusions supported by those measured values.

The run tracker and results document must never treat a missing run as a zero score. Missing, failed, running, trained, partially evaluated, and complete are distinct statuses.

## 11. Acceptance criteria

The implementation is accepted when:

- the notebook contains no MNIST, convolutional autoencoder, DeepSMOTE, Kaggle credential-writing, or synthetic fallback code;
- all four controlled methods are implemented and pass self-tests;
- no mixup representation/target pairing mismatch is possible through the public augmentation interface;
- `SMOKE` completes top to bottom on Kaggle with real D4RL data and simulator interaction;
- interrupted training/evaluation can resume without duplicating completed runs or episodes;
- `FULL_BENCHMARK` enumerates exactly 225 training runs;
- checkpoints, histories, manifests, episode results, summaries, and figures follow the persistence layout;
- every reported aggregate includes its sample count and uncertainty;
- the notebook clearly distinguishes measured results from hypotheses and does not contain fabricated benchmark values;
- every executable notebook section has its management Markdown cell and the opening dashboard is present;
- `README.md`, experiment protocol, Kaggle runbook, run tracker, and results template exist and agree with the frozen configuration in the notebook;
- run status is derived from artifacts and missing runs are never summarized as zero-valued results.

## 12. Out of scope

- IQL, CQL, TD3+BC, or other offline-RL algorithms.
- Pixel observations or image augmentation.
- Dynamics-model learning.
- AntMaze, Adroit, Kitchen, or PyBullet benchmarks.
- Hyperparameter sweeps intended to tune each method separately.
- Claims about real-robot transfer without real-robot experiments.
