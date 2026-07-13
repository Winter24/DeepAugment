#!/usr/bin/env python3
"""Build the separate local k-NN latent mixup Kaggle notebook."""

from pathlib import Path

from build_kaggle_notebook import build


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "latent-space-state-action-local-knn-mixup.ipynb"


if __name__ == "__main__":
    build(output=OUTPUT, include_local_knn=True)
