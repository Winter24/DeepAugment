import json
import hashlib
import runpy
from pathlib import Path


NOTEBOOK = Path("Latent State-Action Mixup.ipynb")


def notebook_text():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])


def test_notebook_has_management_markdown_and_no_legacy_topics():
    text = notebook_text()
    for heading in [
        "Experiment Dashboard", "Configuration", "D4RL Data",
        "Models and Augmentation", "Training and Resume", "Evaluation", "Reporting",
    ]:
        assert heading in text
    for forbidden in [
        "MNIST", "DeepSMOTE", "Convolutional Autoencoder",
        "make_synthetic_dataset", "KAGGLE_USERNAME", "KAGGLE_KEY",
    ]:
        assert forbidden not in text


def test_notebook_embeds_all_five_methods_and_225_run_assertion():
    text = notebook_text()
    for method in ["vanilla_bc", "noise_bc", "input_mixup_bc", "latent_mixup_bc", "local_latent_mixup_bc"]:
        assert method in text
    assert "assert len(runs) == 225" in text


def test_notebook_uses_python312_compatible_backend_and_honest_metric_label():
    text = notebook_text()
    assert "gymnasium[mujoco]" in text
    assert "modern_simulator_compatibility_score" in text
    for forbidden in ("mujoco-py", "gym==0.23.1", "Farama-Foundation/d4rl.git", "numpy<2"):
        assert forbidden not in text


def test_notebook_displays_all_saved_result_figures_inline():
    text = notebook_text()
    assert "Display Saved Result Figures" in text
    assert "from IPython.display import Image, display" in text
    for name in ("clean_scores.png", "robustness.png", "dropoff.png", "validation_loss.png"):
        assert name in text


def test_notebook_has_incremental_three_method_pilot_cells_and_final_plots():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = notebook_text()
    for heading in (
        "Incremental Pilot Comparison",
        "Pilot Method: Noise BC",
        "Pilot Method: Input Mixup BC",
        "Pilot Method: Latent Mixup BC",
        "Pilot Comparison Plots",
    ):
        assert heading in text
    assert 'run_pilot_method("noise_bc")' in text
    assert 'run_pilot_method("input_mixup_bc")' in text
    assert 'run_pilot_method("latent_mixup_bc")' in text
    assert "pilot_config = replace(config, single_method=method)" in text
    assert "Pilot requires RUN_MODE='SINGLE_RUN'" in text


def test_notebook_has_resumable_seed_1_to_4_cells_and_aggregate_plots():
    text = notebook_text()
    assert "Multi-Seed Pilot" in text
    assert "def run_method_seed(method, seed):" in text
    for seed in (1, 2, 3, 4):
        assert f"Pilot Seed {seed}" in text
        assert f"seed_{seed}_results = run_seed({seed})" in text
    assert "Multi-Seed Aggregate Plots" in text
    assert "completed_seed_count" in text
    assert "SEEDS = [0, 1, 2, 3, 4]" in text
    for forbidden in ("numpy==2.2.6", "pandas==2.2.3", "--force-reinstall", "RESTART_REQUIRED"):
        assert forbidden not in text


def test_each_executable_section_is_preceded_by_management_markdown():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code" and "SECTION:" in "".join(cell.get("source", [])):
            assert index > 0
            previous = "".join(nb["cells"][index - 1].get("source", []))
            for label in ("Purpose", "Inputs", "Outputs", "Modes", "Cost", "Recovery"):
                assert label in previous


def test_writefile_cell_magic_is_always_the_first_line():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    writefile_cells = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        if "%%writefile" in source:
            writefile_cells.append(source)
            assert source.startswith("%%writefile ")
    assert writefile_cells


def test_notebook_generation_is_byte_deterministic():
    build = runpy.run_path("scripts/build_kaggle_notebook.py")["build"]
    build()
    first = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    build()
    second = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    assert first == second
