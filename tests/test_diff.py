"""Tests for the diff engine."""

import pytest

from rag_diff.core.diff import diff_runs, DiffReport, CaseDiff, TriageLabel
from rag_diff.models import CaseResult, JudgeVerdict, RunSnapshot


def _snap(run_id: str, results: list[CaseResult]) -> RunSnapshot:
    return RunSnapshot(
        run_id=run_id,
        timestamp="2026-04-10T10:00:00",
        adapter="test:ask",
        testset_path="test.json",
        results=results,
    )


def _case(
    query: str,
    answer: str = "answer",
    hashes: list[str] | None = None,
    latency: float = 100.0,
    tokens: int | None = None,
    verdicts: list[JudgeVerdict] | None = None,
) -> CaseResult:
    return CaseResult(
        query=query,
        answer=answer,
        contexts=[],
        context_hashes=hashes or ["h1"],
        latency_ms=latency,
        tokens_used=tokens,
        judge_verdicts=verdicts,
    )


def _verdict(dim: str, passed: bool, score: float = 1.0) -> JudgeVerdict:
    return JudgeVerdict(dimension=dim, passed=passed, score=score, reason="test")


class TestDiffRunsBasic:
    def test_identical_runs_no_regressions(self):
        base = _snap("base", [_case("q1"), _case("q2")])
        curr = _snap("curr", [_case("q1"), _case("q2")])
        report = diff_runs(base, curr)

        assert isinstance(report, DiffReport)
        assert report.regressions == 0
        assert report.total_cases == 2

    def test_unmatched_cases_are_counted(self):
        base = _snap("base", [_case("q1"), _case("q2")])
        curr = _snap("curr", [_case("q1"), _case("q3")])
        report = diff_runs(base, curr)
        assert report.total_cases == 1  # only q1 matched
        assert len(report.added_cases) == 1
        assert len(report.removed_cases) == 1

    def test_empty_runs(self):
        report = diff_runs(_snap("base", []), _snap("curr", []))
        assert report.total_cases == 0
        assert report.regressions == 0


class TestDiffTriageWithJudge:
    def test_retrieval_regression(self):
        """Judge got worse AND contexts changed → retrieval issue."""
        base = _snap("base", [_case(
            "q1", hashes=["h1", "h2"],
            verdicts=[_verdict("faithfulness", True), _verdict("relevancy", True)],
        )])
        curr = _snap("curr", [_case(
            "q1", hashes=["h1", "h3"],  # h2 lost, h3 new
            verdicts=[_verdict("faithfulness", False), _verdict("relevancy", True)],
        )])
        report = diff_runs(base, curr)
        assert report.regressions == 1
        diff = report.case_diffs[0]
        assert diff.triage == TriageLabel.RETRIEVAL

    def test_model_regression(self):
        """Judge got worse but contexts are identical → model issue."""
        base = _snap("base", [_case(
            "q1", hashes=["h1"],
            verdicts=[_verdict("faithfulness", True), _verdict("relevancy", True)],
        )])
        curr = _snap("curr", [_case(
            "q1", hashes=["h1"],  # same contexts
            verdicts=[_verdict("faithfulness", False), _verdict("relevancy", True)],
        )])
        report = diff_runs(base, curr)
        assert report.regressions == 1
        assert report.case_diffs[0].triage == TriageLabel.MODEL

    def test_improvement_detected(self):
        """Judge improved → status is improvement."""
        base = _snap("base", [_case(
            "q1", verdicts=[_verdict("faithfulness", False), _verdict("relevancy", True)],
        )])
        curr = _snap("curr", [_case(
            "q1", verdicts=[_verdict("faithfulness", True), _verdict("relevancy", True)],
        )])
        report = diff_runs(base, curr)
        assert report.improvements == 1
        assert report.case_diffs[0].status == "improvement"


class TestDiffTriageWithoutJudge:
    def test_context_change_surfaces_as_changed(self):
        """No judge results, but context hashes changed → status 'changed'."""
        base = _snap("base", [_case("q1", hashes=["h1", "h2"])])
        curr = _snap("curr", [_case("q1", hashes=["h1", "h3"])])
        report = diff_runs(base, curr)
        diff = report.case_diffs[0]
        assert diff.status == "changed"
        assert diff.triage == TriageLabel.UNKNOWN
        assert diff.contexts_lost == ["h2"]
        assert diff.contexts_new == ["h3"]
        assert report.changed == 1

    def test_identical_no_judge(self):
        base = _snap("base", [_case("q1", hashes=["h1"])])
        curr = _snap("curr", [_case("q1", hashes=["h1"])])
        report = diff_runs(base, curr)
        assert report.case_diffs[0].status == "unchanged"
        assert report.changed == 0


class TestDiffMetrics:
    def test_latency_delta(self):
        base = _snap("base", [_case("q1", latency=100.0)])
        curr = _snap("curr", [_case("q1", latency=300.0)])
        report = diff_runs(base, curr)
        diff = report.case_diffs[0]
        assert diff.latency_before == 100.0
        assert diff.latency_after == 300.0
        assert diff.latency_delta_pct == pytest.approx(200.0)

    def test_token_delta(self):
        base = _snap("base", [_case("q1", tokens=100)])
        curr = _snap("curr", [_case("q1", tokens=400)])
        report = diff_runs(base, curr)
        diff = report.case_diffs[0]
        assert diff.tokens_before == 100
        assert diff.tokens_after == 400


class TestDiffContextDelta:
    def test_lost_and_new_contexts(self):
        base = _snap("base", [_case("q1", hashes=["a", "b", "c"])])
        curr = _snap("curr", [_case("q1", hashes=["b", "c", "d"])])
        report = diff_runs(base, curr)
        diff = report.case_diffs[0]
        assert diff.contexts_lost == ["a"]
        assert diff.contexts_new == ["d"]

    def test_no_context_change(self):
        base = _snap("base", [_case("q1", hashes=["a", "b"])])
        curr = _snap("curr", [_case("q1", hashes=["a", "b"])])
        report = diff_runs(base, curr)
        diff = report.case_diffs[0]
        assert diff.contexts_lost == []
        assert diff.contexts_new == []
