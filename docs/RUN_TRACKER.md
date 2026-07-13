# Run Tracker

**Protocol:** the expected matrix contains 180 `(dataset, method, seed)` rows.

This file is a human-readable snapshot generated from manifests and episode CSV. The artifact files—not manual checkboxes—are the source of truth.

## Status vocabulary

- `missing`: no durable training manifest and no evaluation data.
- `failed`: a captured run failure.
- `trained`: valid checkpoint/history/manifest, no evaluation.
- `partially evaluated`: some but not all required noise levels/episodes exist.
- `complete`: training and required evaluation coverage are durable.

Missing values remain blank/NaN and are never converted to zero reward.

## Current snapshot

No measured Kaggle runs have been imported yet. Generate this table from `build_run_tracker(...)` after the first SMOKE run.
