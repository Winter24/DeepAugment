from pathlib import Path


def test_required_management_docs_exist_without_fake_results():
    paths = [
        Path("README.md"),
        Path("docs/EXPERIMENT_PROTOCOL.md"),
        Path("docs/KAGGLE_RUNBOOK.md"),
        Path("docs/RUN_TRACKER.md"),
        Path("docs/RESULTS_TEMPLATE.md"),
    ]
    for path in paths:
        assert path.exists()
        assert path.read_text(encoding="utf-8").strip()
    assert "Measured result" in Path("docs/RESULTS_TEMPLATE.md").read_text(encoding="utf-8")
    assert "missing" in Path("docs/RUN_TRACKER.md").read_text(encoding="utf-8").lower()


def test_runbook_documents_all_modes_and_cross_session_resume():
    text = Path("docs/KAGGLE_RUNBOOK.md").read_text(encoding="utf-8")
    for term in ("SMOKE", "SINGLE_RUN", "FULL_BENCHMARK", "Kaggle Dataset", "RESUME_INPUT_ROOT"):
        assert term in text
