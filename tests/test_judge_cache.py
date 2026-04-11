"""Tests for the judge cache module."""

import pytest

from rag_diff.storage.judge_cache import JudgeCache, compute_cache_key


class TestComputeCacheKey:
    def test_deterministic(self):
        key1 = compute_cache_key("q", "a", ["h1", "h2"])
        key2 = compute_cache_key("q", "a", ["h1", "h2"])
        assert key1 == key2

    def test_different_query(self):
        k1 = compute_cache_key("q1", "a", ["h1"])
        k2 = compute_cache_key("q2", "a", ["h1"])
        assert k1 != k2

    def test_different_answer(self):
        k1 = compute_cache_key("q", "a1", ["h1"])
        k2 = compute_cache_key("q", "a2", ["h1"])
        assert k1 != k2

    def test_different_contexts(self):
        k1 = compute_cache_key("q", "a", ["h1"])
        k2 = compute_cache_key("q", "a", ["h2"])
        assert k1 != k2

    def test_context_order_independent(self):
        k1 = compute_cache_key("q", "a", ["h2", "h1"])
        k2 = compute_cache_key("q", "a", ["h1", "h2"])
        assert k1 == k2

    def test_returns_sha256_hex(self):
        key = compute_cache_key("q", "a", [])
        assert len(key) == 64


@pytest.fixture
def cache(tmp_path):
    return JudgeCache(db_path=str(tmp_path / "judge_cache.db"))


class TestJudgeCacheGetSet:
    def test_miss_returns_none(self, cache):
        assert cache.get("nonexistent_key", "v1.0") is None

    def test_round_trip(self, cache):
        results = [{"dimension": "faithfulness", "passed": True, "score": 1.0, "reason": "ok"}]
        cache.set("key1", "v1.0", results)
        got = cache.get("key1", "v1.0")
        assert got == results

    def test_version_mismatch_returns_none(self, cache):
        results = [{"dimension": "faithfulness", "passed": True, "score": 1.0, "reason": "ok"}]
        cache.set("key1", "v1.0", results)
        assert cache.get("key1", "v2.0") is None

    def test_overwrite_same_key(self, cache):
        r1 = [{"dimension": "faithfulness", "passed": True, "score": 1.0, "reason": "first"}]
        r2 = [{"dimension": "faithfulness", "passed": False, "score": 0.0, "reason": "second"}]
        cache.set("key1", "v1.0", r1)
        cache.set("key1", "v1.0", r2)
        got = cache.get("key1", "v1.0")
        assert got[0]["reason"] == "second"

    def test_multiple_keys(self, cache):
        r1 = [{"dimension": "faithfulness", "passed": True, "score": 1.0, "reason": "a"}]
        r2 = [{"dimension": "relevancy", "passed": False, "score": 0.3, "reason": "b"}]
        cache.set("key1", "v1.0", r1)
        cache.set("key2", "v1.0", r2)
        assert cache.get("key1", "v1.0") == r1
        assert cache.get("key2", "v1.0") == r2

    def test_multi_dimension_results(self, cache):
        results = [
            {"dimension": "faithfulness", "passed": True, "score": 0.9, "reason": "grounded"},
            {"dimension": "relevancy", "passed": False, "score": 0.2, "reason": "off-topic"},
        ]
        cache.set("key1", "v1.0", results)
        got = cache.get("key1", "v1.0")
        assert len(got) == 2
        assert got[0]["dimension"] == "faithfulness"
        assert got[1]["dimension"] == "relevancy"


class TestJudgeCacheStats:
    def test_stats_empty(self, cache):
        stats = cache.stats()
        assert stats["total"] == 0

    def test_stats_after_inserts(self, cache):
        cache.set("k1", "v1.0", [])
        cache.set("k2", "v1.0", [])
        stats = cache.stats()
        assert stats["total"] == 2
