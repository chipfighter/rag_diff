"""Tests for the run manager module."""

import pytest

from rag_diff.models import CaseResult, RunSnapshot
from rag_diff.storage.run_manager import RunManager


@pytest.fixture
def manager(tmp_path):
    return RunManager(base_dir=tmp_path / ".ragdiff")


@pytest.fixture
def sample_snapshot():
    return RunSnapshot(
        run_id="run_20260410_1000_abc12345",
        timestamp="2026-04-10T10:00:00",
        adapter="my_adapter:ask",
        testset_path="tests/test.json",
        results=[
            CaseResult(
                query="What is X?",
                answer="X is Y",
                contexts=["context text"],
                context_hashes=["abcdef1234567890" * 4],
                latency_ms=150.5,
            )
        ],
    )


class TestRunManagerSave:
    def test_creates_run_directory(self, manager, sample_snapshot):
        manager.save(sample_snapshot)
        run_dir = manager.base_dir / "runs" / sample_snapshot.run_id
        assert run_dir.exists()
        assert (run_dir / "snapshot.json").exists()

    def test_updates_head_pointer(self, manager, sample_snapshot):
        manager.save(sample_snapshot)
        assert manager.get_head() == sample_snapshot.run_id

    def test_head_tracks_latest(self, manager, sample_snapshot):
        manager.save(sample_snapshot)
        second = sample_snapshot.model_copy(update={"run_id": "run_20260410_1100_def45678"})
        manager.save(second)
        assert manager.get_head() == "run_20260410_1100_def45678"


class TestRunManagerLoad:
    def test_round_trip(self, manager, sample_snapshot):
        manager.save(sample_snapshot)
        loaded = manager.load(sample_snapshot.run_id)
        assert loaded.run_id == sample_snapshot.run_id
        assert len(loaded.results) == 1
        assert loaded.results[0].query == "What is X?"
        assert loaded.results[0].answer == "X is Y"
        assert loaded.adapter == "my_adapter:ask"

    def test_nonexistent_run_raises(self, manager):
        with pytest.raises(FileNotFoundError):
            manager.load("nonexistent_run_id")


class TestRunManagerHead:
    def test_no_runs_returns_none(self, manager):
        assert manager.get_head() is None


class TestRunManagerListRuns:
    def test_empty_initially(self, manager):
        assert manager.list_runs() == []

    def test_lists_saved_runs(self, manager, sample_snapshot):
        manager.save(sample_snapshot)
        second = sample_snapshot.model_copy(update={"run_id": "run_20260410_1100_def45678"})
        manager.save(second)
        runs = manager.list_runs()
        assert len(runs) == 2
        assert sample_snapshot.run_id in runs
        assert "run_20260410_1100_def45678" in runs

    def test_list_is_sorted(self, manager, sample_snapshot):
        manager.save(sample_snapshot)
        second = sample_snapshot.model_copy(update={"run_id": "run_20260410_1100_def45678"})
        manager.save(second)
        runs = manager.list_runs()
        assert runs == sorted(runs)


class TestGenerateRunId:
    def test_format(self, manager):
        run_id = manager.generate_run_id()
        assert run_id.startswith("run_")
        parts = run_id.split("_")
        assert len(parts) == 4
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 4  # HHMM
        assert len(parts[3]) == 8  # short uuid

    def test_unique(self, manager):
        ids = {manager.generate_run_id() for _ in range(10)}
        assert len(ids) == 10
