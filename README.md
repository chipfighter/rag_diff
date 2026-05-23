# RAG-Diff

**Regression testing and diagnostic diffing for RAG pipelines.**

RAG-Diff is a CLI tool that helps you answer three questions every time you change your RAG system:

1. **Did I break it?** — Snapshot-based regression testing
2. **Was it retrieval or the model?** — Heuristic triage
3. **Show me what changed.** — Context-level diffs with LOST/NEW tracking

Think of it as `pytest` for RAG — lightweight, fast, developer-first.

## Install

```bash
pip install rag-diff

# With LLM-as-Judge support (optional):
pip install rag-diff[openai]
```

**Requirements:** Python 3.10+

## Quick Start

### 1. Initialize a sample project

```bash
rag-diff init
```

This creates `sample_adapter.py` and `sample_testset.json`.

### 2. Run your first test

```bash
rag-diff run --testset sample_testset.json --target sample_adapter.py:ask
```

### 3. Make a change and run again

```bash
# Edit your adapter, change retrieval, swap models, etc.
rag-diff run --testset sample_testset.json --target sample_adapter.py:ask
```

On the second run, RAG-Diff automatically compares against the previous run and shows you what changed:

```
+- Diff: run_20260410_1000_abc12345 -> run_20260410_1015_def67890 -+
| 1 regression(s) | 1 changed | 8 unchanged                       |
+------------------------------------------------------------------+

  REGRESSION How is overtime calculated?
    Triage: Suspected Retrieval Issue
    Latency: 120ms -> 350ms (+192%)
    Judge:
      faithfulness: PASS (1.0) -> FAIL (0.2)
        Reason: Answer mentions "3x pay" but new contexts lack this policy.
    Contexts:
      - LOST 4a8b2c1d9e3f7a05...
      + NEW  f1e2d3c4b5a69078...
```

## Writing an Adapter

An adapter is an async (or sync) Python function that takes a query and returns a dict:

```python
# Minimal — just answer and contexts
async def ask(query: str) -> dict:
    result = my_rag_pipeline(query)
    return {
        "answer": result.answer,
        "contexts": [chunk.text for chunk in result.chunks],
    }

# Recommended — structured contexts with IDs and scores
async def ask(query: str) -> dict:
    result = my_rag_pipeline(query)
    return {
        "answer": result.answer,
        "contexts": [
            {"id": c.id, "text": c.text, "score": c.score}
            for c in result.chunks
        ],
    }
```

RAG-Diff auto-generates SHA-256 hashes for plain text contexts, so you don't need to provide IDs.

## Test Set Format

A JSON array of test cases:

```json
[
    {"query": "What is the refund policy?"},
    {"query": "How many vacation days?", "expected_answer": "15 days after 5 years"},
    {"query": "Overtime rules", "metadata": {"category": "hr", "priority": "high"}}
]
```

- `query` (required): The question to ask your RAG system
- `expected_answer` (optional): Golden answer for higher judge accuracy
- `metadata` (optional): Arbitrary tags for your own tracking

## CLI Reference

### `rag-diff run`

Run the test set and record a snapshot.

```bash
rag-diff run --testset tests.json --target my_adapter.py:ask [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--testset` | (required) | Path to test set JSON file |
| `--target` | (required) | Adapter in `module:function` format |
| `--output-dir` | `.ragdiff` | Directory for run data |
| `--concurrency` | `5` | Max concurrent adapter calls |
| `--compare-to` | (auto: HEAD) | Specific run ID to diff against |
| `--judge / --no-judge` | `--judge` | Enable/disable LLM evaluation |
| `--judge-model` | `gpt-4o-mini` | Model for LLM-as-Judge |
| `--regressions-only` | off | Only show regressions in output |
| `--dump` | off | Export diff report as JSON |

### `rag-diff init [DIRECTORY]`

Create sample adapter and test set files.

### `rag-diff list`

List all recorded runs with HEAD marker.

## How It Works

### Architecture

```
rag-diff run --testset test.json --target adapter.py:ask
     |
     v
  cli.py --> loader.py (dynamic import)
     |
     v
  runner.py --> adapter(query) --> CaseResult
     |              |
     |              v
     |       context_hash.py (SHA-256 fingerprint)
     |
     v
  judge.py --> LLM (faithfulness + relevancy)
     |              |
     |              v
     |       judge_cache.py (SQLite, version-aware)
     |
     v
  run_manager.py --> .ragdiff/runs/{run_id}/snapshot.json
     |
     v
  diff.py --> heuristic triage (retrieval vs model)
     |
     v
  console.py --> Rich terminal output
```

### Heuristic Triage

When a regression is detected, RAG-Diff classifies the root cause:

| Scenario | Triage | What it means |
|----------|--------|---------------|
| Judge failed + contexts changed | **Retrieval Issue** | Your retriever returned different documents |
| Judge failed + contexts same | **Model Issue** | Same context, worse answer — model/prompt problem |
| No judge + contexts changed | **Changed** | Structural change detected, enable judge for classification |

### Judge Cache

LLM-as-Judge calls are expensive. RAG-Diff caches every verdict in a local SQLite database, keyed by `SHA-256(query + answer + sorted_context_hashes)` and scoped to the judge prompt version. If inputs haven't changed, the cached verdict is reused — typical cache hit rates are **80%+** during iterative development.

### Judge Prompt Version

Judge prompts are locked to `v1.0` to prevent evaluation drift. When prompts change in a future release, the version bumps and cached results are automatically invalidated.

## Data Layout

```
.ragdiff/
├── runs/
│   ├── run_20260410_1000_abc12345/
│   │   └── snapshot.json          # Full run results
│   └── run_20260410_1015_def67890/
│       └── snapshot.json
├── cache/
│   └── judge_cache_v1.db          # SQLite judge cache
├── dumps/
│   └── diff_20260410_1015.json    # --dump output
└── HEAD.json                      # Points to latest run
```

Add `.ragdiff/` to your `.gitignore`.

## Using with CI

```yaml
# GitHub Actions example
- name: RAG regression test
  run: |
    pip install rag-diff[openai]
    rag-diff run --testset tests/rag_tests.json --target src/adapter.py:ask
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Roadmap

- **Phase 2**: Agentic route diff, HTML report exporter
- **Phase 3**: Custom judge injection, multi-dimensional failure classification

## License

MIT
