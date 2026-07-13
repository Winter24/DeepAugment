# Results

## Protocol

The frozen comparison and interpretation rules are defined in `EXPERIMENT_PROTOCOL.md`. Any deviation must be recorded before presenting results.

## Measured result

No Kaggle benchmark result has been imported. Populate this section only from generated `episode_results.csv` and `summary.csv`.

Required outputs:

- clean `modern_simulator_compatibility_score` per dataset/method as mean ± standard deviation with seed count;
- robustness curves across all observation-noise levels;
- relative drop-off from clean score;
- paired latent-mixup-versus-baseline differences and bootstrap confidence intervals;
- failed and missing run counts.

## Interpretation

Do not conclude superiority until measurements exist and the frozen interpretation rule is satisfied. Discuss task-specific regressions, simulator compatibility, behavior-cloning limitations, observation-noise scope, and absence of real-robot evidence.

Never call the compatibility metric a standard D4RL v2 normalized score. Training data is D4RL v2, while online evaluation uses Gymnasium MuJoCo v5.
