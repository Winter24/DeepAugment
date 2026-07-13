import json
from pathlib import Path


NOTEBOOK = Path("latent-space-state-action-local-knn-mixup.ipynb")


def notebook_text():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])


def test_local_knn_notebook_is_separate_and_contains_incremental_seed_cells():
    assert NOTEBOOK.exists()
    text = notebook_text()
    assert "In-batch k-NN Latent State–Action Mixup" in text
    assert "local_latent_mixup_bc" in text
    assert "action_threshold" in text
    for seed in range(5):
        assert f"Local k-NN Seed {seed}" in text
        assert f'run_method_seed("local_latent_mixup_bc", {seed})' in text
    assert "Five-Method Aggregate Comparison" in text


def test_local_knn_notebook_keeps_cell_magics_valid():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        if "%%writefile" in source:
            assert source.startswith("%%writefile ")


def test_local_knn_notebook_is_extension_only_and_skips_old_training_cells():
    text = notebook_text()
    assert "Extension notebook: baseline training skipped" in text
    assert "Extension notebook: baseline evaluation skipped" in text
    assert "Old method pilot skipped; restored artifacts are reused" in text
    assert "Old multi-seed training skipped; restored artifacts are reused" in text
