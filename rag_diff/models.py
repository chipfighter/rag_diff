"""Data models for RAG-Diff."""

from typing import Optional, Union

from pydantic import BaseModel


class TestCase(BaseModel):
    """A single test case in a test set."""

    query: str
    expected_answer: Optional[str] = None
    metadata: Optional[dict] = None


class JudgeVerdict(BaseModel):
    """Result of a single judge evaluation dimension."""

    dimension: str  # "faithfulness" | "relevancy"
    passed: bool
    score: float  # 0.0 to 1.0
    reason: str


class CaseResult(BaseModel):
    """Result of running a single test case through the adapter."""

    query: str
    answer: str
    contexts: list[Union[str, dict]]
    context_hashes: list[str]
    latency_ms: float
    tokens_used: Optional[int] = None
    expected_answer: Optional[str] = None
    metadata: Optional[dict] = None
    judge_verdicts: Optional[list[JudgeVerdict]] = None


class RunSnapshot(BaseModel):
    """A complete snapshot of a test run."""

    run_id: str
    timestamp: str
    adapter: str
    testset_path: str
    results: list[CaseResult]
    judge_prompt_version: str = "v1.0"
    config: Optional[dict] = None
