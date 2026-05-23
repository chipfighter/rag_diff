## 1. Core Positioning (The Vision)

**"The pytest of the RAG world" — from test harness to debugger**

RAG-Diff focuses on solving the "black-box anxiety" and "high testing cost" of iterating on RAG systems. By recording behavior snapshots, freezing the evaluation criteria, caching evaluation results, and providing code-level comparison tracking, it lets developers answer three core questions at extremely low cost:

1. Did this change break the system? (Regression Test)
2. Did retrieval get worse, or did the model get dumber? (Heuristic Triage)
3. Where exactly did it rot? Show me the before/after text. (Context Diff)

### 1.1 Competitive Differentiation

| Existing tool       | What it does                       | What it doesn't do                          |
|---------------------|------------------------------------|---------------------------------------------|
| Ragas / DeepEval    | Produces scores (Faithfulness 0.82)| Doesn't tell you "what regressed vs. last time" |
| Arize Phoenix       | Production-grade trace observability| Doesn't provide fast dev-time regression comparison |
| LangSmith           | Full-chain tracing                 | Ops-heavy, SaaS-bound, not lightweight enough |
| promptfoo           | Prompt A/B evaluation              | Doesn't focus on RAG retrieval-vs-generation triage |

RAG-Diff fills the gap of **"fast regression detection + root-cause localization inside the developer's iteration loop."**

------

## 2. Core System Architecture

### 2.1 Stupid Simple Adapter

Built on a "graceful degradation" design philosophy: it does not force users to modify the return values of their existing business code. The system tracks context internally via automatic **SHA-256** hashing (compared to MD5, this avoids collision risk and passes security audits).

Python

```
# Core contract: just make it run; plain-text degradation is supported
async def ask_rag(query: str) -> dict:
    return {
        "answer": "...",
        "contexts": ["text1", "text2"]  # the system automatically computes a SHA-256 hash and generates virtual IDs for tracking
    }

# Recommended form (structured data):
async def ask_rag(query: str) -> dict:
    return {
        "answer": "...",
        "contexts": [{"id": "doc_12_chunk_1", "text": "...", "score": 0.88}],
        "route_path": ["policy_db"]  # [reserved / optional] routing label
    }
```

### 2.2 Evaluation Cache Engine (Judge Cache) — **P0-level core**

To fully eliminate the token cost and time consumption of LLM evaluation, the MVP ships with a mandatory caching mechanism.

- **Cache Key**: `SHA-256(Query + Answer + sorted Context Hashes)`
- **Hit logic**: As long as a case's input and output state haven't changed, directly reuse the Judge result from the previous (`compare-to` target) run. The Judge LLM is only called on a cache miss.

### 2.2.1 Multi-Dimension Judging

The MVP ships with two core evaluation dimensions to avoid blind spots from a single dimension:

- **Faithfulness**: Is the Answer grounded in the given Contexts? Detects hallucination.
- **Answer Relevancy**: Does the Answer actually address the Query? Detects off-topic answers / refusals.

### 2.2.2 Test Set Schema

Supports an optional `expected_answer` field; having a Golden Answer substantially improves Judge accuracy:

```json
[
  {"query": "How is holiday overtime pay calculated?"},
  {"query": "How are annual leave days determined?", "expected_answer": "Based on years of service: 5/10/15 days", "metadata": {"category": "hr"}}
]
```

### 2.3 Version Freezing & State Isolation

Abandons the traditional "overwrite-style baseline" in favor of Git-like snapshot isolation:

- **Judge version locking**: Hard-codes `JUDGE_PROMPT_VERSION = "v1.0"` to guard against evaluation-criteria drift.
- **Isolated workspace**: Each Run generates its own UUID directory.

### 2.4 Root-Cause Heuristic Triage

When an evaluation regression is detected, automatically perform a binary diagnosis:

- 🔍 **Suspected Retrieval Issue**: The Judge verdict got worse, AND this run's Context IDs/Hashes differ from the baseline.
- 🧠 **Suspected Model Issue**: The Judge verdict got worse, BUT the Contexts list is identical to the baseline.

------

## 3. Developer Experience Design (Killer DX)

### 3.1 CLI Command Routing

Real developer ergonomics, supporting a "time machine" and "precision strike."

Bash

```
# 0. Initialize project (generate .ragdiff/ + sample testset + sample adapter)
$ rag-diff init

# 1. Basic run (compares against HEAD by default)
$ rag-diff run --testset test.json --target adapter:ask

# 2. Precise comparison (time-machine mode)
$ rag-diff run --compare-to run_20260410_1000_UUID

# 3. Efficient debug mode (only run regressed cases)
$ rag-diff run --regressions-only

# 4. Minimal SRE & cost control
$ rag-diff run --concurrency 5 --sample 20
```

### 3.2 Terminal Output & SRE Monitoring (Terminal UI)

Combines minimal performance monitoring (Latency & Token) to expose core problems directly in the terminal.

Plaintext

```
🔴 [Regression] Q12: How is holiday overtime pay calculated?
[Triage]: 🔍 Suspected Retrieval Issue

--- [Metrics Diff] ---
Latency: 1.2s -> 3.5s 🔴 (+191%)
Tokens:  1024 -> 4500 🔴 (Cost Alert)

--- [Judge Diff] ---
Previous (v1.0): ✅ Grounded
Current  (v1.0): ❌ Has Unsupported Claims
Reason: The answer mentions "3x pay," but the new Contexts lack any related provision.

--- [Contexts Diff] ---
- [LOST] id: hr_policy_v1_chunk_42 (or the corresponding plain-text Hash)
+ [NEW]  id: faq_leave_chunk_5

💡 Developer tip: use the --dump command to export the full snapshot for IDE diff comparison.
```

### 3.3 IDE-Friendly Scene Dump (JSON Dump)

Skip complex HTML rendering and emit a structured "scene" directly, so developers can use the built-in diff tools of VSCode / Cursor to inspect long text.

Bash

```
# Run a single case precisely and export the context snapshot
$ rag-diff run -k "Q12" --dump

# Artifact: .ragdiff/dumps/Q12_diff_20260410.json
```

------

## 4. Project Structure

Plaintext

```
rag-diff/
├── rag_diff/
│   ├── cli.py              # Typer command routing/parsing
│   ├── core/
│   │   ├── runner.py       # asyncio concurrency pool + minimal latency/token monitoring
│   │   ├── judge.py        # JUDGE_VERSION freezing + cache-hit logic
│   │   ├── diff.py         # diff diagnosis engine with Heuristic Triage
│   │   └── context_hash.py # SHA-256 converter for plain-text degradation
│   ├── storage/
│   │   └── run_manager.py  # maintains the .ragdiff/runs/ structure and HEAD pointer
│   └── utils/
│       └── console.py      # Rich terminal UI rendering
├── .ragdiff/               # runtime-generated directory (add to .gitignore)
│   ├── runs/
│   │   └── run_20260410_1000_UUID/
│   ├── cache/
│   │   └── judge_cache_v1.db # local SQLite or JSON Lines cache file
│   ├── dumps/              # debug scenes exported via --dump
│   └── HEAD.json           # records the latest_run pointer
```

------

------

## Appendix: Evolution Roadmap (Future Roadmap)

The following features have been explicitly cut from the MVP (v0.1) stage to ensure the core experience ships fast. They are directions for future versions:

### Phase 2: Observability & Experience Upgrade (SRE & UX Plus)

- **Agentic Route Diff (route tracing)**: Read the optional `route_path` field from the Adapter. When neither retrieval nor generation is anomalous but the result still regresses, diagnose it as `[Triage]: 🔀 Suspected Routing Issue`.
- **HTML Report Exporter**: Provide an `--export-html` command that generates a GitHub-style side-by-side highlighted page, giving a more intuitive reading experience for ultra-long contexts (200K+).
- In `Phase 2`, RAG-Diff could also add **Diff tracking** for these two dimensions:
  - **Stance Diff:** If the baseline was "in favor" and after this change it becomes "against," raise a high-priority alert even if Faithfulness scores are all high (Critical Logic Shift).
  - **Citation Coordinate Diff:** Record the character offset of a citation in the source text. If the offset jumps from `[100, 200]` to `[5000, 5200]`, it often indicates the model has experienced serious citation drift (Citation Drift).

### Phase 3: Judge Decoupling

- **Custom Judge Injection**: Allow developers to bypass the built-in cloud LLM API evaluation and register a local scoring function (e.g., a locally deployed, dedicated Llama-3-8B evaluation model) to satisfy on-premise compliance requirements for sensitive data.
- **Multi-Dimensional Failure Classification**: Expand from a single "hallucination/refusal" axis to finer-grained Triage diagnoses such as tone violations, format errors, and content redundancy.
