from pathlib import Path

from research_agent.cli import chroma_dir_for_run


def test_chroma_dir_for_run_uses_run_id_when_experiment_id_is_missing():
    path = chroma_dir_for_run(Path("data/chroma"), "run-123")

    assert path == Path("data/chroma/run-123")


def test_chroma_dir_for_run_uses_slugified_experiment_id():
    path = chroma_dir_for_run(Path("data/chroma"), "run-123", "My Experiment 01")

    assert path == Path("data/chroma/my-experiment-01")
