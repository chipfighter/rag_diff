"""LLM-as-Judge evaluation module.

Evaluates RAG outputs on two dimensions — Faithfulness and Answer Relevancy —
with version-locked prompts and SQLite-backed caching to minimize LLM cost.
"""

import json
import re
from typing import Awaitable, Callable, Optional

from rag_diff.models import JudgeVerdict
from rag_diff.storage.judge_cache import JudgeCache, compute_cache_key

JUDGE_PROMPT_VERSION = "v1.0"

# Type alias: takes a prompt string, returns the LLM response string
LLMCaller = Callable[[str], Awaitable[str]]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_faithfulness_prompt(query: str, answer: str, contexts: list[str]) -> str:
    """Build the faithfulness evaluation prompt."""
    ctx_block = "\n\n".join(
        f"[Context {i + 1}]\n{c}" for i, c in enumerate(contexts)
    )
    return (
        "You are an impartial judge evaluating whether an AI assistant's "
        "answer is grounded in the provided context.\n\n"
        f"Query: {query}\n\n"
        f"Context:\n{ctx_block}\n\n"
        f"Answer: {answer}\n\n"
        "Evaluate whether every claim in the answer is supported by the context.\n\n"
        "Respond with EXACTLY this JSON format (no other text):\n"
        '{"passed": true/false, "score": 0.0-1.0, "reason": "brief explanation"}\n\n'
        "- passed: true if the answer is fully supported by the context\n"
        "- score: 1.0 for fully grounded, 0.0 for completely ungrounded\n"
        "- reason: one-sentence explanation"
    )


def build_relevancy_prompt(
    query: str,
    answer: str,
    expected_answer: Optional[str] = None,
) -> str:
    """Build the answer relevancy evaluation prompt."""
    expected_section = ""
    if expected_answer:
        expected_section = f"\nExpected answer (reference): {expected_answer}\n"

    return (
        "You are an impartial judge evaluating whether an AI assistant's "
        "answer addresses the user's query.\n\n"
        f"Query: {query}\n\n"
        f"Answer: {answer}\n"
        f"{expected_section}\n"
        "Evaluate whether the answer directly and helpfully addresses the query.\n\n"
        "Respond with EXACTLY this JSON format (no other text):\n"
        '{"passed": true/false, "score": 0.0-1.0, "reason": "brief explanation"}\n\n'
        "- passed: true if the answer addresses the query\n"
        "- score: 1.0 for perfectly relevant, 0.0 for completely irrelevant\n"
        "- reason: one-sentence explanation"
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_judge_response(raw: str, dimension: str) -> JudgeVerdict:
    """Parse an LLM judge response into a JudgeVerdict.

    Handles raw JSON or JSON wrapped in markdown code blocks.
    Returns a fail verdict on parse errors rather than crashing.
    """
    cleaned = raw.strip()

    # Strip markdown code fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()

    try:
        data = json.loads(cleaned)
        return JudgeVerdict(
            dimension=dimension,
            passed=bool(data["passed"]),
            score=float(data["score"]),
            reason=str(data["reason"]),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return JudgeVerdict(
            dimension=dimension,
            passed=False,
            score=0.0,
            reason=f"Failed to parse judge response: {raw[:200]}",
        )


# ---------------------------------------------------------------------------
# Extracting context text
# ---------------------------------------------------------------------------

def _extract_context_texts(contexts: list) -> list[str]:
    """Extract plain text strings from a mixed context list."""
    texts = []
    for ctx in contexts:
        if isinstance(ctx, str):
            texts.append(ctx)
        elif isinstance(ctx, dict):
            texts.append(ctx.get("text", ctx.get("id", str(ctx))))
        else:
            texts.append(str(ctx))
    return texts


# ---------------------------------------------------------------------------
# Main judge entry point
# ---------------------------------------------------------------------------

async def judge_case(
    query: str,
    answer: str,
    contexts: list,
    context_hashes: list[str],
    *,
    cache: JudgeCache,
    llm_caller: Optional[LLMCaller] = None,
    expected_answer: Optional[str] = None,
) -> Optional[list[JudgeVerdict]]:
    """Judge a single case on faithfulness and relevancy.

    Returns None if no llm_caller is provided (judging disabled).
    Uses the cache to skip LLM calls when inputs haven't changed.
    """
    if llm_caller is None:
        return None

    cache_key = compute_cache_key(query, answer, context_hashes)

    # Check cache
    cached = cache.get(cache_key, JUDGE_PROMPT_VERSION)
    if cached is not None:
        return [JudgeVerdict(**v) for v in cached]

    # Cache miss — call LLM for each dimension
    context_texts = _extract_context_texts(contexts)

    faithfulness_prompt = build_faithfulness_prompt(query, answer, context_texts)
    relevancy_prompt = build_relevancy_prompt(query, answer, expected_answer)

    faith_raw = await llm_caller(faithfulness_prompt)
    relev_raw = await llm_caller(relevancy_prompt)

    verdicts = [
        parse_judge_response(faith_raw, "faithfulness"),
        parse_judge_response(relev_raw, "relevancy"),
    ]

    # Persist to cache
    cache.set(
        cache_key,
        JUDGE_PROMPT_VERSION,
        [v.model_dump() for v in verdicts],
    )

    return verdicts
