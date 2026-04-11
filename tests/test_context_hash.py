"""Tests for the context_hash module."""

from rag_diff.core.context_hash import hash_context, hash_context_list, normalize_text


class TestNormalizeText:
    def test_strips_leading_trailing_whitespace(self):
        assert normalize_text("  hello world  ") == "hello world"

    def test_collapses_internal_whitespace(self):
        assert normalize_text("hello   world") == "hello world"

    def test_handles_newlines_and_tabs(self):
        assert normalize_text("hello\n\tworld") == "hello world"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_already_clean(self):
        assert normalize_text("clean text") == "clean text"


class TestHashContext:
    def test_plain_text_returns_sha256_hex(self):
        result = hash_context("hello world")
        assert len(result) == 64  # SHA-256 hex digest length

    def test_deterministic(self):
        assert hash_context("hello world") == hash_context("hello world")

    def test_different_text_produces_different_hash(self):
        assert hash_context("hello") != hash_context("world")

    def test_whitespace_is_normalized_before_hashing(self):
        assert hash_context("hello  world") == hash_context("hello world")
        assert hash_context("  hello world  ") == hash_context("hello world")

    def test_structured_context_uses_text_field(self):
        ctx = {"id": "doc_1", "text": "hello world", "score": 0.9}
        assert hash_context(ctx) == hash_context("hello world")

    def test_structured_context_falls_back_to_id(self):
        ctx = {"id": "doc_1"}
        result = hash_context(ctx)
        assert len(result) == 64
        assert result == hash_context("doc_1")

    def test_structured_context_empty_dict(self):
        result = hash_context({})
        assert len(result) == 64


class TestHashContextList:
    def test_returns_sorted_hashes(self):
        hashes = hash_context_list(["banana", "apple"])
        assert hashes == sorted(hashes)

    def test_empty_list(self):
        assert hash_context_list([]) == []

    def test_single_item(self):
        hashes = hash_context_list(["only one"])
        assert len(hashes) == 1
        assert hashes[0] == hash_context("only one")

    def test_mixed_plain_and_structured(self):
        contexts = ["plain text", {"id": "doc_1", "text": "structured text"}]
        hashes = hash_context_list(contexts)
        assert len(hashes) == 2
        assert all(len(h) == 64 for h in hashes)

    def test_deterministic_regardless_of_input_order(self):
        list_a = hash_context_list(["b text", "a text"])
        list_b = hash_context_list(["a text", "b text"])
        assert list_a == list_b
