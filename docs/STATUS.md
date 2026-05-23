# RAG-Diff — Project Situation

_Last updated: 2026-05-23_

## TL;DR

The core idea of RAG-Diff is **proven**. We built a real RAG pipeline, pointed
the tool at it with a real LLM (DeepSeek), and it correctly told two kinds of
failure apart: a broken retriever vs. a broken model. This was a proof of
concept — feasibility is now answered ("yes, it works"). It is **not yet
decided** whether this becomes an MVP-to-ship or stays an incremental
experiment.

## What RAG-Diff is

A local-first CLI tool — "pytest for RAG." Every time you change a RAG system it
answers three questions:

1. Did I break it? (snapshot-based regression testing)
2. Was it retrieval or the model? (heuristic triage)
3. What changed? (context-level diff with LOST/NEW tracking)

Stack: `typer` + `rich` + `pydantic`, a SQLite judge cache, JSON snapshots.

## Current state

- **v0.1 tool**: code-complete, 118 unit tests passing.
- **Validated end-to-end (2026-05-23)** against a real RAG pipeline + real LLM.
  Before this date the only test subject was a dummy adapter that echoed the
  query — the tool had never been proven on anything real.

## What was built in the PoC

- **`examples/real_rag/`** — a real RAG pipeline used as the subject under test:
  markdown HR corpus → local `sentence-transformers` embeddings → NumPy
  cosine-similarity retrieval → DeepSeek answer. It exposes "regression knobs"
  so you can break it on purpose (see its `README.md`).
- **`.env` support in the tool** (the only change to the core): added
  `python-dotenv` and a `load_dotenv()` call in `cli.py`. Because the judge
  uses the OpenAI SDK, pointing it at DeepSeek (or any OpenAI-compatible API)
  now just means editing `.env` — no code change. Keys are bring-your-own (no
  SaaS); `.env` is gitignored.

## What was proven

| Scenario          | How we broke it        | RAG-Diff's verdict          |
|-------------------|------------------------|-----------------------------|
| Good baseline     | nothing                | mostly PASS                 |
| Broken retrieval  | `RAG_RETRIEVAL=bad`    | **Suspected Retrieval Issue** (contexts LOST/NEW; answers became "I don't know") |
| Broken generation | `RAG_PROMPT_MODE=lazy` | **Suspected Model Issue** (faithfulness FAIL; contexts unchanged) |

The same tool, same corpus, separated the two root causes correctly.

## Findings & caveats

1. **Fewer retrieved chunks ≠ worse retrieval.** `RAG_TOPK=1` caused *zero*
   regressions because each answer fits in its single top chunk. A decisive
   break required returning the *wrong* chunks.
2. **The LLM judge is a bit noisy.** A good answer was once scored "fail," and a
   degraded run showed a spurious "improvement." Useful, but not 100% reliable.

Full details: [`drafts/poc_findings.md`](../drafts/poc_findings.md).

## Not done yet (deliberately deferred)

Retrieval metrics (Recall@k / MRR / nDCG), ground-truth labels, PyPI publish,
HTML reports, and unrelated doc-vs-code gaps (latency-in-triage, `--sample`/`-k`).

## Open questions for next time

- Make the judge more reliable (golden-answer anchoring, multi-sample), **or**
- Add deterministic retrieval metrics (Recall@k) — no LLM noise, and the deepest
  RAG-learning step. Both findings above point this way.
- Decide the project's nature: MVP-to-ship vs. incremental experiment.

## How to run (quick)

```powershell
pip install -e ".[openai,dev]"
pip install -r examples/real_rag/requirements.txt
# Copy env.example to .env and add a DeepSeek key (OPENAI_API_KEY + OPENAI_BASE_URL)
rag-diff run --testset examples/real_rag/testset.json --target examples/real_rag/adapter.py:ask --judge-model deepseek-chat
```

See [`examples/real_rag/README.md`](../examples/real_rag/README.md) for the full
break-it-on-purpose walkthrough.

## Key files

```
rag_diff/                   the tool (cli.py, core/, storage/, utils/)
examples/real_rag/          real RAG pipeline (PoC subject under test)
e2e_test/                   dummy adapters (plumbing test)
tests/                      118 unit tests
drafts/project_draft.md     original design doc (translated to English)
drafts/poc_findings.md      PoC results
docs/STATUS.md              this file
agentic_content/HANDOFF.md  context handoff for another AI agent
env.example                 .env template
```
