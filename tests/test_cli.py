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
        assert "not found" in result.output.lower() or result.exit_code == 1

    def test_successful_run(self, sample_testset, sample_adapter, tmp_path):
        output_dir = str(tmp_path / ".ragdiff")
        result = runner.invoke(
            app,
            [
                "run",
                "--testset", str(sample_testset),
                "--target", f"{sample_adapter}:ask",
                "--output-dir", output_dir,
            ],
        )
        assert result.exit_code == 0
        assert "Run complete" in result.output or "complete" in result.output.lower()

        # Verify HEAD.json was created
        head_path = tmp_path / ".ragdiff" / "HEAD.json"
        assert head_path.exists()
        head_data = json.loads(head_path.read_text())
        assert head_data["latest_run"].startswith("run_")

    def test_run_creates_snapshot(self, sample_testset, sample_adapter, tmp_path):
        output_dir = str(tmp_path / ".ragdiff")
        runner.invoke(
            app,
            [
                "run",
                "--testset", str(sample_testset),
                "--target", f"{sample_adapter}:ask",
                "--output-dir", output_dir,
            ],
        )
        runs_dir = tmp_path / ".ragdiff" / "runs"
        assert runs_dir.exists()
        run_dirs = list(runs_dir.iterdir())
        assert len(run_dirs) == 1
        assert (run_dirs[0] / "snapshot.json").exists()

    def test_run_snapshot_content(self, sample_testset, sample_adapter, tmp_path):
        output_dir = str(tmp_path / ".ragdiff")
        runner.invoke(
            app,
            [
                "run",
                "--testset", str(sample_testset),
                "--target", f"{sample_adapter}:ask",
                "--output-dir", output_dir,
            ],
        )
        runs_dir = tmp_path / ".ragdiff" / "runs"
        run_dir = list(runs_dir.iterdir())[0]
        snapshot = json.loads((run_dir / "snapshot.json").read_text())
        assert len(snapshot["results"]) == 2
        assert snapshot["results"][0]["query"] == "What is RAG?"
        assert "Answer to:" in snapshot["results"][0]["answer"]

    def test_concurrency_option(self, sample_testset, sample_adapter, tmp_path):
        output_dir = str(tmp_path / ".ragdiff")
        result = runner.invoke(
            app,
            [
                "run",
                "--testset", str(sample_testset),
                "--target", f"{sample_adapter}:ask",
                "--output-dir", output_dir,
                "--concurrency", "1",
            ],
        )
        assert result.exit_code == 0


class TestCliNoArgs:
    def test_no_command_shows_help(self):
        result = runner.invoke(app, [])
        # Typer exits with code 0 or 2 when showing help via no_args_is_help
        assert result.exit_code in (0, 2)
        assert "usage" in result.output.lower()
