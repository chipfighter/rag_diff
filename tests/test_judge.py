"""Tests for the judge module."""

import json

import pytest

from rag_diff.core.judge import (
    JUDGE_PROMPT_VERSION,
    judge_case,
    build_faithfulness_prompt,
    build_relevancy_prompt,
    parse_judge_response,
)
from rag_diff.models import JudgeVerdict
from rag_diff.storage.judge_cache import JudgeCache


# --- Prompt building ---


class TestBuildFaithfulnessPrompt:
    def test_contains_query(self):
        prompt = build_faithfulness_prompt("my query", "my answer", ["ctx1"])
        assert "my query" in prompt

    def test_contains_answer(self):
        prompt = build_faithfulness_prompt("q", "my answer", ["ctx1"])
        assert "my answer" in prompt

    def test_contains_contexts(self):
        prompt = build_faithfulness_prompt("q", "a", ["context one", "context two"])
        assert "context one" in prompt
        assert "context two" in prompt


class TestBuildRelevancyPrompt:
    def test_contains_query_and_answer(self):
        prompt = build_relevancy_prompt("my query", "my answer")
        assert "my query" in prompt
        assert "my answer" in prompt

    def test_includes_expected_answer_when_provided(self):
        prompt = build_relevancy_prompt("q", "a", expected_answer="golden")
        assert "golden" in prompt

    def test_no_expected_answer_section_when_none(self):
        prompt = build_relevancy_prompt("q", "a")
        assert "Expected" not in prompt


# --- Response parsing ---


class TestParseJudgeResponse:
    def test_valid_json(self):
        raw = '{"passed": true, "score": 0.95, "reason": "all good"}'
        result = parse_judge_response(raw, "faithfulness")
        assert isinstance(result, JudgeVerdict)
        assert result.passed is True
        assert result.score == 0.95
        assert result.dimension == "faithfulness"

    def test_json_in_markdown_code_block(self):
        raw = '```json\n{"passed": false, "score": 0.1, "reason": "hallucinated"}\n```'
        result = parse_judge_response(raw, "faithfulness")
        assert result.passed is False

    def test_malformed_json_returns_fail_verdict(self):
        raw = "This is not JSON at all"
        result = parse_judge_response(raw, "relevancy")
        assert result.passed is False
        assert result.score == 0.0
        assert "parse" in result.reason.lower() or "fail" in result.reason.lower()


# --- Full judge_case flow ---


@pytest.fixture
def cache(tmp_path):
    return JudgeCache(db_path=str(tmp_path / "cache.db"))


class TestJudgeCase:
    @pytest.mark.asyncio
    async def test_calls_llm_and_returns_verdicts(self, cache):
        call_count = 0

        async def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return json.dumps({"passed": True, "score": 0.9, "reason": "ok"})

        verdicts = await judge_case(
            query="What is X?",
            answer="X is Y",
            contexts=["Y is defined as ..."],
            context_hashes=["abc"],
            cache=cache,
            llm_caller=mock_llm,
        )

        assert len(verdicts) == 2  # faithfulness + relevancy
        assert all(isinstance(v, JudgeVerdict) for v in verdicts)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cache_hit_skips_llm(self, cache):
        call_count = 0

        async def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return json.dumps({"passed": True, "score": 1.0, "reason": "cached"})

        # First call — populates cache
        await judge_case(
            query="q", answer="a", contexts=["c"], context_hashes=["h1"],
            cache=cache, llm_caller=mock_llm,
        )
        first_count = call_count

        # Second call — same inputs, should hit cache
        verdicts = await judge_case(
            query="q", answer="a", contexts=["c"], context_hashes=["h1"],
            cache=cache, llm_caller=mock_llm,
        )
        assert call_count == first_count  # no new LLM calls
        assert len(verdicts) == 2

    @pytest.mark.asyncio
    async def test_cache_miss_on_different_answer(self, cache):
        call_count = 0

        async def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return json.dumps({"passed": True, "score": 0.8, "reason": "ok"})

        await judge_case(
            query="q", answer="a1", contexts=["c"], context_hashes=["h1"],
            cache=cache, llm_caller=mock_llm,
        )
        first_count = call_count

        await judge_case(
            query="q", answer="a2", contexts=["c"], context_hashes=["h1"],
            cache=cache, llm_caller=mock_llm,
        )
        assert call_count == first_count + 2  # 2 new calls

    @pytest.mark.asyncio
    async def test_no_llm_caller_skips_judging(self, cache):
        verdicts = await judge_case(
            query="q", answer="a", contexts=["c"], context_hashes=["h1"],
            cache=cache, llm_caller=None,
        )
        assert verdicts is None

    @pytest.mark.asyncio
    async def test_passes_expected_answer_to_relevancy(self, cache):
        prompts_seen = []

        async def capture_llm(prompt: str) -> str:
            prompts_seen.append(prompt)
            return json.dumps({"passed": True, "score": 1.0, "reason": "ok"})

        await judge_case(
            query="q", answer="a", contexts=["c"], context_hashes=["h1"],
            expected_answer="golden answer",
            cache=cache, llm_caller=capture_llm,
        )
        # The relevancy prompt (second call) should contain the expected answer
        assert any("golden answer" in p for p in prompts_seen)

    @pytest.mark.asyncio
    async def test_version_is_v1(self):
        assert JUDGE_PROMPT_VERSION == "v1.0"
