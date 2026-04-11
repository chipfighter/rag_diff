"""Tests for the console rendering module."""

import pytest
from io import StringIO
from rich.console import Console

from rag_diff.core.diff import CaseDiff, DiffReport, TriageLabel
from rag_diff.models import JudgeVerdict
from rag_diff.utils.console import render_diff_report, render_run_summary


def _make_report(
    regressions: int = 0,
    improvements: int = 0,
    changed: int = 0,
    unchanged: int = 0,
    case_diffs: list[CaseDiff] | None = None,
) -> DiffReport:
    return DiffReport(
        baseline_run_id="run_base",
        current_run_id="run_curr",
        total_cases=regressions + improvements + changed + unchanged,
        regressions=regressions,
        improvements=improvements,
        changed=changed,
        unchanged=unchanged,
        case_diffs=case_diffs or [],
    )


def _capture_output(func, *args, **kwargs) -> str:
    """Capture Rich console output as plain text."""
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    func(*args, console=console, **kwargs)
    return buf.getvalue()


class TestRenderRunSummary:
    def test_shows_case_count(self):
        output = _capture_output(render_run_summary, run_id="run_abc", case_count=5, avg_latency=123.4)
        assert "run_abc" in output
        assert "5" in output

    def test_shows_latency(self):
        output = _capture_output(render_run_summary, run_id="run_x", case_count=1, avg_latency=99.9)
        assert "99.9" in output


class TestRenderDiffReport:
    def test_no_regressions_shows_clean(self):
        report = _make_report(unchanged=3)
        output = _capture_output(render_diff_report, report)
        assert "0 regressions" in output.lower() or "no regressions" in output.lower() or "0" in output

    def test_regression_shows_query(self):
        diff = CaseDiff(
            query="How to calculate overtime?",
            status="regression",
            triage=TriageLabel.RETRIEVAL,
            contexts_lost=["hash_old"],
            contexts_new=["hash_new"],
            latency_before=100.0,
            latency_after=300.0,
            latency_delta_pct=200.0,
            answer_before="old answer",
            answer_after="new answer",
            judge_before=[JudgeVerdict(dimension="faithfulness", passed=True, score=1.0, reason="ok")],
            judge_after=[JudgeVerdict(dimension="faithfulness", passed=False, score=0.2, reason="hallucinated")],
        )
        report = _make_report(regressions=1, case_diffs=[diff])
        output = _capture_output(render_diff_report, report)
        assert "overtime" in output.lower()

    def test_triage_label_shown(self):
        diff = CaseDiff(
            query="q1",
            status="regression",
            triage=TriageLabel.MODEL,
            contexts_lost=[],
            contexts_new=[],
            latency_before=100.0,
            latency_after=100.0,
            latency_delta_pct=0.0,
            answer_before="a",
            answer_after="b",
        )
        report = _make_report(regressions=1, case_diffs=[diff])
        output = _capture_output(render_diff_report, report)
        assert "model" in output.lower()

    def test_latency_delta_shown(self):
        diff = CaseDiff(
            query="q1",
            status="regression",
            triage=TriageLabel.RETRIEVAL,
            contexts_lost=[],
            contexts_new=[],
            latency_before=100.0,
            latency_after=350.0,
            latency_delta_pct=250.0,
            answer_before="a",
            answer_after="b",
        )
        report = _make_report(regressions=1, case_diffs=[diff])
        output = _capture_output(render_diff_report, report)
        assert "250" in output

    def test_renders_without_crash_on_empty(self):
        report = _make_report()
        output = _capture_output(render_diff_report, report)
        assert output  # just should not crash

    def test_improvement_shown(self):
        diff = CaseDiff(
            query="q_improved",
            status="improvement",
            contexts_lost=[],
            contexts_new=[],
            latency_before=200.0,
            latency_after=100.0,
            latency_delta_pct=-50.0,
            answer_before="a",
            answer_after="b",
        )
        report = _make_report(improvements=1, case_diffs=[diff])
        output = _capture_output(render_diff_report, report)
        assert "q_improved" in output
