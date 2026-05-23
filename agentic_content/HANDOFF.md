# Agent Handoff — RAG-Diff

_For a Claude agent picking up this repo. Written 2026-05-23 by the previous
agent. Read this first, then `docs/STATUS.md`._

## Your immediate mode: DISCUSS, don't code

The user wants to **talk and analyze the next step first**, not jump into
implementation. Do NOT start writing code until they explicitly say so. The last
work (a proof of concept) is finished; the next move is a decision to be made
*together*.

## How to work with this user (important)

- **English is their second language.** Keep your writing **simple and short**.
  Avoid dense paragraphs and jargon; explain a term in plain words if you must
  use it.
- **Don't give long multi-option menus.** They get overwhelmed by 4-way choices.
  Summarize, then **recommend ONE option** with a one-line reason. Offer a simple
  yes/no to confirm.
- **Discuss before coding. Never code without explicit permission.** They like to
  align on scope first, then say "go."
- **Mindset is "PoC / learning."** Their PRIMARY goal is to *learn RAG deeply*
  (especially how to build benchmarks and measure RAG performance). SECONDARY: a
  small open-source tool they'd use themselves at work (enterprise RAG). No SaaS.
- They may drop a Chinese word inline when a concept is hard — that's fine, work
  with it.

(There is also a persistent memory for this project at the Claude memory dir;
these same facts live there as `user-profile`, `feedback-communicate-simply`,
and `project-rag-diff`.)

## What this repo is

RAG-Diff = a local-first CLI, "pytest for RAG." It snapshots each RAG run, diffs
two runs at the context level (LOST/NEW chunk hashes), and does retrieval-vs-
model **triage**. Stack: typer + rich + pydantic, SQLite judge cache, JSON
snapshots. 118 unit tests pass.

## State as of this handoff

**The PoC SUCCEEDED.** We built a real RAG pipeline and proved the tool's core
hypothesis on real data with a real LLM (DeepSeek):

- Break retrieval (`RAG_RETRIEVAL=bad`) → tool says **"Suspected Retrieval Issue."**
- Break generation (`RAG_PROMPT_MODE=lazy`, retrieval intact) → **"Suspected Model Issue."**

Two honest findings: (1) fewer retrieved chunks ≠ worse retrieval (had to break
retrieval by returning *wrong* chunks, not *fewer*); (2) the LLM judge has
noticeable variance. See `drafts/poc_findings.md`.

## What's in the repo (key files)

```
rag_diff/                   the tool: cli.py, core/{runner,judge,diff,context_hash}, storage/, utils/
examples/real_rag/          NEW real RAG pipeline (the PoC subject)
  pipeline.py               chunk -> local embeddings -> NumPy cosine retrieval -> DeepSeek
  adapter.py                ask(query); knobs: RAG_TOPK, RAG_RETRIEVAL, RAG_PROMPT_MODE
  corpus/                   fake HR handbook (markdown)
  testset.json, README.md
e2e_test/                   dummy adapters for a no-LLM plumbing check
tests/                      118 unit tests
drafts/project_draft.md     original design doc (translated from Chinese)
drafts/poc_findings.md      PoC results + caveats
docs/STATUS.md              full human-readable situation
env.example                 copy to .env, add a DeepSeek key
```

## Only core change made

Added `python-dotenv` + `load_dotenv()` in `cli.py`. The judge already uses the
OpenAI SDK, so DeepSeek works via `.env` (`OPENAI_API_KEY` + `OPENAI_BASE_URL=
https://api.deepseek.com`, run with `--judge-model deepseek-chat`). The tool core
stays light; the example's heavy deps (sentence-transformers/torch) are isolated
in `examples/real_rag/requirements.txt`.

## How to run it

```powershell
pip install -e ".[openai,dev]"
pip install -r examples/real_rag/requirements.txt
# .env must exist at repo root with a DeepSeek key (see env.example). .env is gitignored.
rag-diff run --testset examples/real_rag/testset.json --target examples/real_rag/adapter.py:ask --judge-model deepseek-chat
```

Retrieval-only sanity check (no API key): `python examples/real_rag/pipeline.py`.

## Where to pick up — the open decision

The PoC answered "can it be done?" → yes. The likely next step (to be decided
WITH the user, not for them) is one of:

1. **Make the judge more reliable** — golden-answer anchoring, multiple samples.
2. **Add deterministic retrieval metrics** (Recall@k / MRR) — no LLM noise, needs
   ground-truth labels in the test set. This is also the richest RAG-learning
   step, and both PoC findings point toward it.

Also still undecided on purpose: is this an MVP-to-ship or an incremental
experiment? Don't assume — ask.

Start by reading `docs/STATUS.md` and `drafts/poc_findings.md`, then have a
simple, short conversation with the user about which direction they want.
