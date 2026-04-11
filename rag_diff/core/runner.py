"""Test runner for executing RAG adapters against test cases.

Manages async concurrent execution with bounded parallelism,
captures latency, and normalizes adapter responses into CaseResults.
"""

import asyncio
import inspect
import time
from typing import Awaitable, Callable, Union

from rag_diff.core.context_hash import hash_context_list
from rag_diff.models import CaseResult, TestCase


async def run_single_case(
    adapter: Callable[[str], Union[dict, Awaitable[dict]]],
    case: TestCase,
) -> CaseResult:
    """Run a single test case through the adapter and collect results."""
    start = time.perf_counter()

    if inspect.iscoroutinefunction(adapter):
        response = await adapter(case.query)
    else:
        response = adapter(case.query)

    elapsed_ms = (time.perf_counter() - start) * 1000

    answer = response["answer"]
    contexts = response.get("contexts", [])
    context_hashes = hash_context_list(contexts)

    return CaseResult(
        query=case.query,
        answer=answer,
        contexts=contexts,
        context_hashes=context_hashes,
        latency_ms=round(elapsed_ms, 2),
        tokens_used=response.get("tokens_used"),
        expected_answer=case.expected_answer,
        metadata=case.metadata,
    )


async def run_testset(
    adapter: Callable[[str], Union[dict, Awaitable[dict]]],
    cases: list[TestCase],
    concurrency: int = 5,
) -> list[CaseResult]:
    """Run all test cases with bounded concurrency, preserving input order."""
    if not cases:
        return []

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_run(case: TestCase) -> CaseResult:
        async with semaphore:
            return await run_single_case(adapter, case)

    tasks = [bounded_run(case) for case in cases]
    return list(await asyncio.gather(*tasks))
