"""Tests for the runner module."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from rag_diff.core.runner import run_single_case, run_testset
from rag_diff.models import CaseResult, TestCase


class TestRunSingleCase:
    @pytest.mark.asyncio
    async def test_basic_execution(self):
        adapter = AsyncMock(
            return_value={"answer": "42", "contexts": ["ctx1", "ctx2"]}
        )
        case = TestCase(query="What is the answer?")

        result = await run_single_case(adapter, case)

        assert isinstance(result, CaseResult)
        assert result.query == "What is the answer?"
        assert result.answer == "42"
        assert result.latency_ms > 0
        assert len(result.context_hashes) == 2
        adapter.assert_called_once_with("What is the answer?")

    @pytest.mark.asyncio
    async def test_structured_contexts(self):
        adapter = AsyncMock(
            return_value={
                "answer": "structured",
                "contexts": [{"id": "doc_1", "text": "hello", "score": 0.9}],
            }
        )
        case = TestCase(query="test")

        result = await run_single_case(adapter, case)

        assert len(result.contexts) == 1
        assert len(result.context_hashes) == 1

    @pytest.mark.asyncio
    async def test_preserves_expected_answer(self):
        adapter = AsyncMock(return_value={"answer": "actual", "contexts": []})
        case = TestCase(query="q", expected_answer="expected")

        result = await run_single_case(adapter, case)

        assert result.expected_answer == "expected"

    @pytest.mark.asyncio
    async def test_preserves_metadata(self):
        adapter = AsyncMock(return_value={"answer": "a", "contexts": []})
        case = TestCase(query="q", metadata={"category": "hr"})

        result = await run_single_case(adapter, case)

        assert result.metadata == {"category": "hr"}

    @pytest.mark.asyncio
    async def test_captures_tokens_used(self):
        adapter = AsyncMock(
            return_value={"answer": "a", "contexts": [], "tokens_used": 150}
        )
        case = TestCase(query="q")

        result = await run_single_case(adapter, case)

        assert result.tokens_used == 150

    @pytest.mark.asyncio
    async def test_adapter_error_propagates(self):
        adapter = AsyncMock(side_effect=RuntimeError("API error"))
        case = TestCase(query="q")

        with pytest.raises(RuntimeError, match="API error"):
            await run_single_case(adapter, case)


class TestRunTestset:
    @pytest.mark.asyncio
    async def test_runs_all_cases(self):
        adapter = AsyncMock(return_value={"answer": "a", "contexts": ["c"]})
        cases = [TestCase(query=f"q{i}") for i in range(3)]

        results = await run_testset(adapter, cases, concurrency=2)

        assert len(results) == 3
        assert adapter.call_count == 3

    @pytest.mark.asyncio
    async def test_respects_concurrency_limit(self):
        max_concurrent = 0
        current_concurrent = 0

        async def slow_adapter(query):
            nonlocal max_concurrent, current_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.05)
            current_concurrent -= 1
            return {"answer": "a", "contexts": []}

        cases = [TestCase(query=f"q{i}") for i in range(6)]

        results = await run_testset(slow_adapter, cases, concurrency=2)

        assert len(results) == 6
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_empty_testset(self):
        adapter = AsyncMock()

        results = await run_testset(adapter, [], concurrency=5)

        assert results == []
        adapter.assert_not_called()

    @pytest.mark.asyncio
    async def test_preserves_case_order(self):
        async def echo_adapter(query):
            return {"answer": query, "contexts": []}

        cases = [TestCase(query=f"q{i}") for i in range(5)]

        results = await run_testset(echo_adapter, cases, concurrency=3)

        assert [r.query for r in results] == [f"q{i}" for i in range(5)]
