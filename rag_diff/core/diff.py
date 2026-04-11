"""Diff engine with heuristic triage for comparing RAG run snapshots.

Compares two RunSnapshots case-by-case and classifies regressions as
retrieval-related or model-related based on context hash changes and
judge verdict deltas.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel

from rag_diff.models import CaseResult, JudgeVerdict, RunSnapshot


class TriageLabel(str, Enum):
    """Root-cause triage classification."""

    RETRIEVAL = "retrieval"  # contexts changed + judge regressed
    MODEL = "model"          # contexts same + judge regressed
    UNKNOWN = "unknown"      # no judge data available


class CaseDiff(BaseModel):
    """Diff result for a single matched test case."""

    query: str
    status: str  # "regression" | "improvement" | "unchanged"
    triage: Optional[TriageLabel] = None

    # Judge deltas
    judge_before: Optional[list[JudgeVerdict]] = None
    judge_after: Optional[list[JudgeVerdict]] = None

    # Context deltas
    contexts_lost: list[str]  # hashes present in baseline but not current
    contexts_new: list[str]   # hashes present in current but not baseline

    # Metrics
    latency_before: float
    latency_after: float
    latency_delta_pct: float
    tokens_before: Optional[int] = None
    tokens_after: Optional[int] = None

    # Raw answers for text diff
    answer_before: str = ""
    answer_after: str = ""


class DiffReport(BaseModel):
    """Aggregate diff report comparing two runs."""

    baseline_run_id: str
    current_run_id: str
    total_cases: int
    regressions: int
    improvements: int
    changed: int  # contexts changed but no judge data to classify
    unchanged: int
    case_diffs: list[CaseDiff]
    added_cases: list[str] = []    # queries in current but not baseline
    removed_cases: list[str] = []  # queries in baseline but not current


def _judge_pass_count(verdicts: Optional[list[JudgeVerdict]]) -> Optional[int]:
    """Count how many dimensions passed, or None if no verdicts."""
    if verdicts is None:
        return None
    return sum(1 for v in verdicts if v.passed)


def _compute_latency_delta_pct(before: float, after: float) -> float:
    if before == 0:
        return 0.0 if after == 0 else 100.0
    return ((after - before) / before) * 100.0


def _diff_case(base: CaseResult, curr: CaseResult) -> CaseDiff:
    """Compute the diff between a baseline and current case result."""
    base_hashes = set(base.context_hashes)
    curr_hashes = set(curr.context_hashes)
    contexts_lost = sorted(base_hashes - curr_hashes)
    contexts_new = sorted(curr_hashes - base_hashes)
    contexts_changed = bool(contexts_lost or contexts_new)

    base_passes = _judge_pass_count(base.judge_verdicts)
    curr_passes = _judge_pass_count(curr.judge_verdicts)

    # Determine status and triage
    if base_passes is not None and curr_passes is not None:
        base_total = len(base.judge_verdicts)  # type: ignore[arg-type]
        curr_total = len(curr.judge_verdicts)  # type: ignore[arg-type]
        # Normalize to comparable scores
        base_rate = base_passes / base_total if base_total else 0
        curr_rate = curr_passes / curr_total if curr_total else 0

        if curr_rate < base_rate:
            status = "regression"
            triage = TriageLabel.RETRIEVAL if contexts_changed else TriageLabel.MODEL
        elif curr_rate > base_rate:
            status = "improvement"
            triage = None
        else:
            status = "unchanged"
            triage = None
    else:
        # No judge data — surface structural changes so the user sees them
        if contexts_changed:
            status = "changed"
            triage = TriageLabel.UNKNOWN
        else:
            status = "unchanged"
            triage = None

    return CaseDiff(
        query=base.query,
        status=status,
        triage=triage,
        judge_before=base.judge_verdicts,
        judge_after=curr.judge_verdicts,
        contexts_lost=contexts_lost,
        contexts_new=contexts_new,
        latency_before=base.latency_ms,
        latency_after=curr.latency_ms,
        latency_delta_pct=round(
            _compute_latency_delta_pct(base.latency_ms, curr.latency_ms), 1
        ),
        tokens_before=base.tokens_used,
        tokens_after=curr.tokens_used,
        answer_before=base.answer,
        answer_after=curr.answer,
    )


def diff_runs(baseline: RunSnapshot, current: RunSnapshot) -> DiffReport:
    """Compare two run snapshots and produce a diff report.

    Cases are matched by query string. Unmatched cases are reported
    separately as added/removed.
    """
    base_map = {r.query: r for r in baseline.results}
    curr_map = {r.query: r for r in current.results}

    matched_queries = sorted(set(base_map) & set(curr_map))
    added = sorted(set(curr_map) - set(base_map))
    removed = sorted(set(base_map) - set(curr_map))

    case_diffs = [_diff_case(base_map[q], curr_map[q]) for q in matched_queries]

    regressions = sum(1 for d in case_diffs if d.status == "regression")
    improvements = sum(1 for d in case_diffs if d.status == "improvement")
    changed = sum(1 for d in case_diffs if d.status == "changed")
    unchanged = sum(1 for d in case_diffs if d.status == "unchanged")

    return DiffReport(
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        total_cases=len(matched_queries),
        regressions=regressions,
        improvements=improvements,
        changed=changed,
        unchanged=unchanged,
        case_diffs=case_diffs,
        added_cases=added,
        removed_cases=removed,
    )
