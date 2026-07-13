# Experiment Protocol

**Protocol — frozen before benchmark results are inspected.**

## Matrix

- Environments: Hopper, HalfCheetah, Walker2d.
- Datasets: `medium-v2`, `medium-replay-v2`, `medium-expert-v2` for each environment.
- Methods: `vanilla_bc`, `noise_bc`, `input_mixup_bc`, `latent_mixup_bc`, `local_latent_mixup_bc`.
- Seeds: 0–4, totaling 225 training runs.
- Evaluation: 20 episodes at normalized observation-noise standard deviations `0, .01, .05, .10, .20`.

## Controlled comparison

Every method uses the same 256–256 encoder, 128-dimensional latent, 256–256 policy head, optimizer, batch size, early stopping, trajectory split, initialization seed, and evaluation seeds. State mean/std comes only from training trajectories. Validation contains disjoint trajectories.

For input and random latent mixup, one `lambda ~ Beta(0.2, 0.2)` tensor and one batch permutation are reused for representation and action target. Noise BC adds Gaussian noise with standard deviation `.05` in normalized-state units during training.

Local latent mixup chooses the nearest in-batch latent neighbor whose raw action L2 distance is at most `1.0`. Samples without an eligible neighbor remain unchanged. Neighbor search uses detached tensors; the mixed latent and action share the same neighbor and lambda.

## Metrics and interpretation

Primary outcome for this Kaggle-compatible phase is `modern_simulator_compatibility_score`: Gymnasium MuJoCo v5 return scaled using the published D4RL random/expert references. It is not directly comparable to standard D4RL v2 scores because simulator dynamics differ. Supporting outcomes are raw return, validation MSE, relative clean-to-noisy drop, and paired seed-level bootstrap confidence intervals. Episodes are averaged within seed before uncertainty across seeds is computed.

**Interpretation rule:** latent mixup is not called superior when paired confidence intervals are inconclusive, clean performance materially regresses, or gains disappear at task level. Missing experiments remain missing and are never encoded as score zero.
