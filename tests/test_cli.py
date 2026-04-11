"""Tests for the CLI module."""

import json

import pytest
from typer.testing import CliRunner

from rag_diff.cli import app

runner = CliRunner()


@pytest.fixture
def sample_testset(tmp_path):
    testset_path = tmp_path / "test.json"
    testset_path.write_text(
        json.dumps(
            [
                {"query": "What is RAG?"},
                {"query": "How does retrieval work?", "expected_answer": "By searching..."},
            ]
        )
    )
    return testset_path


@pytest.fixture
def sample_adapter(tmp_path):
    adapter_path = tmp_path / "adapter.py"
    adapter_path.write_text(
        "async def ask(query: str) -> dict:\n"
        "    return {\n"
        "        'answer': f'Answer to: {query}',\n"
        "        'contexts': ['some context'],\n"
        "    }\n"
    )
    return adapter_path


def _run_cmd(sample_testset, sample_adapter, tmp_path, extra_args=None):
    """Helper to invoke the run command with common args."""
    output_dir = str(tmp_path / ".ragdiff")
    args = [
        "run",
        "--testset", str(sample_testset),
        "--target", f"{sample_adapter}:ask",
        "--output-dir", output_dir,
        "--no-judge",
    ]
    if extra_args:
        args.extend(extra_args)
    return runner.invoke(app, args), output_dir


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------


class TestCliRun:
    def test_missing_testset_flag(self):
        result = runner.invoke(app, ["run", "--target", "foo:bar"])
        assert result.exit_code != 0

    def test_missing_target_flag(self, sample_testset):
        result = runner.invoke(app, ["run", "--testset", str(sample_testset)])
        assert result.exit_code != 0

    def test_nonexistent_testset_file(self):
        result = runner.invoke(
            app,
            ["run", "--testset", "/nonexistent/path.json", "--target", "foo:bar"],
        )
        assert result.exit_code != 0

    def test_successful_run(self, sample_testset, sample_adapter, tmp_path):
        result, output_dir = _run_cmd(sample_testset, sample_adapter, tmp_path)
        assert result.exit_code == 0
        assert "complete" in result.output.lower()

        head_path = tmp_path / ".ragdiff" / "HEAD.json"
        assert head_path.exists()
        head_data = json.loads(head_path.read_text())
        assert head_data["latest_run"].startswith("run_")

    def test_run_creates_snapshot(self, sample_testset, sample_adapter, tmp_path):
        _run_cmd(sample_testset, sample_adapter, tmp_path)
        runs_dir = tmp_path / ".ragdiff" / "runs"
        assert runs_dir.exists()
        run_dirs = list(runs_dir.iterdir())
        assert len(run_dirs) == 1
        assert (run_dirs[0] / "snapshot.json").exists()

    def test_run_snapshot_content(self, sample_testset, sample_adapter, tmp_path):
        _run_cmd(sample_testset, sample_adapter, tmp_path)
        runs_dir = tmp_path / ".ragdiff" / "runs"
        run_dir = list(runs_dir.iterdir())[0]
        snapshot = json.loads((run_dir / "snapshot.json").read_text())
        assert len(snapshot["results"]) == 2
        assert snapshot["results"][0]["query"] == "What is RAG?"
        assert "Answer to:" in snapshot["results"][0]["answer"]

    def test_concurrency_option(self, sample_testset, sample_adapter, tmp_path):
        result, _ = _run_cmd(sample_testset, sample_adapter, tmp_path, ["--concurrency", "1"])
        assert result.exit_code == 0


class TestCliRunDiff:
    def test_second_run_shows_diff(self, sample_testset, sample_adapter, tmp_path):
        """Running twice should produce a diff report on the second run."""
        _run_cmd(sample_testset, sample_adapter, tmp_path)
        result, _ = _run_cmd(sample_testset, sample_adapter, tmp_path)
        assert result.exit_code == 0
        # Should show diff summary (0 regressions since same adapter)
        assert "0 regression" in result.output.lower()

    def test_first_run_no_diff(self, sample_testset, sample_adapter, tmp_path):
        """First run should say no previous run to compare."""
        result, _ = _run_cmd(sample_testset, sample_adapter, tmp_path)
        assert result.exit_code == 0
        assert "no previous run" in result.output.lower()

    def test_compare_to_specific_run(self, sample_testset, sample_adapter, tmp_path):
        """--compare-to should load a specific baseline."""
        _run_cmd(sample_testset, sample_adapter, tmp_path)
        # Get the run ID
        head_data = json.loads((tmp_path / ".ragdiff" / "HEAD.json").read_text())
        first_run_id = head_data["latest_run"]

        result, _ = _run_cmd(
            sample_testset, sample_adapter, tmp_path,
            ["--compare-to", first_run_id],
        )
        assert result.exit_code == 0

    def test_compare_to_nonexistent_fails(self, sample_testset, sample_adapter, tmp_path):
        result, _ = _run_cmd(
            sample_testset, sample_adapter, tmp_path,
            ["--compare-to", "nonexistent_run"],
        )
        assert result.exit_code != 0

    def test_dump_flag(self, sample_testset, sample_adapter, tmp_path):
        """--dump should export a JSON diff after two runs."""
        _run_cmd(sample_testset, sample_adapter, tmp_path)
        result, output_dir = _run_cmd(
            sample_testset, sample_adapter, tmp_path,
            ["--dump"],
        )
        assert result.exit_code == 0
        dumps_dir = tmp_path / ".ragdiff" / "dumps"
        assert dumps_dir.exists()
        dump_files = list(dumps_dir.glob("*.json"))
        assert len(dump_files) >= 1


# ---------------------------------------------------------------------------
# init command
# ---------------------------------------------------------------------------


class TestCliInit:
    def test_creates_sample_files(self, tmp_path):
        target = str(tmp_path / "myproject")
        result = runner.invoke(app, ["init", target])
        assert result.exit_code == 0
        assert (tmp_path / "myproject" / "sample_adapter.py").exists()
        assert (tmp_path / "myproject" / "sample_testset.json").exists()

    def test_does_not_overwrite(self, tmp_path):
        target = str(tmp_path)
        (tmp_path / "sample_adapter.py").write_text("existing")
        result = runner.invoke(app, ["init", target])
        assert result.exit_code == 0
        assert "skipped" in result.output.lower()
        assert (tmp_path / "sample_adapter.py").read_text() == "existing"

    def test_shows_get_started(self, tmp_path):
        target = str(tmp_path / "fresh")
        result = runner.invoke(app, ["init", target])
        assert "get started" in result.output.lower()


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------


class TestCliList:
    def test_empty_list(self, tmp_path):
        result = runner.invoke(app, ["list", "--output-dir", str(tmp_path / ".ragdiff")])
        assert result.exit_code == 0
        assert "no runs" in result.output.lower()

    def test_lists_runs(self, sample_testset, sample_adapter, tmp_path):
        _run_cmd(sample_testset, sample_adapter, tmp_path)
        result = runner.invoke(app, ["list", "--output-dir", str(tmp_path / ".ragdiff")])
        assert result.exit_code == 0
        assert "run_" in result.output


# ---------------------------------------------------------------------------
# no command
# ---------------------------------------------------------------------------


class TestCliNoArgs:
    def test_no_command_shows_help(self):
        result = runner.invoke(app, [])
        assert result.exit_code in (0, 2)
        assert "usage" in result.output.lower()
