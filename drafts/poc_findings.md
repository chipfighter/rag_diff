# RAG-Diff PoC — Findings (2026-05-23)

## Question
Does RAG-Diff actually work on a real RAG pipeline, and does its triage
correctly tell "retrieval broke" apart from "the model broke"?

## Setup
- Subject under test: `examples/real_rag/` — a real RAG pipeline
  (markdown HR corpus -> local `sentence-transformers` embeddings ->
  NumPy cosine retrieval -> DeepSeek answer).
- LLM: DeepSeek (`deepseek-chat`) for both generation and the judge,
  via the OpenAI-compatible API + a local `.env`.
- 6-question test set.

## Result: SUCCESS — both failure types diagnosed correctly

| Scenario           | Knob                  | RAG-Diff verdict                                                                 |
|--------------------|-----------------------|---------------------------------------------------------------------------------|
| Good baseline      | (none)                | mostly PASS                                                                      |
| Broken retrieval   | `RAG_RETRIEVAL=bad`   | regressions -> **Suspected Retrieval Issue** (contexts LOST/NEW, answers "I don't know") |
| Broken generation  | `RAG_PROMPT_MODE=lazy`| 4 regressions -> **Suspected Model Issue** (faithfulness FAIL, contexts unchanged) |

The same tool, on the same corpus, correctly separated the two root causes.
This is the core hypothesis of RAG-Diff, and it holds on real data.

## What did NOT work as expected (useful findings)
1. **`RAG_TOPK=1` caused zero regressions.** The corpus is "atomic" — each
   answer lives entirely in its single top chunk — so retrieving fewer chunks
   did not lower answer quality. Lesson: *fewer retrieved docs != worse
   retrieval*. We needed a decisive break (return the least-relevant chunks) to
   simulate a genuinely broken retriever.
2. **The LLM judge has noticeable variance.** In the good baseline, one case
   was scored relevancy FAIL even though the answer was fine; the lazy run then
   showed a spurious "improvement". The judge is useful but not perfectly
   reliable.

## Implications for the next decision
- The core idea is validated -> worth continuing.
- Judge variance argues for: golden-answer-anchored judging, and/or adding
  **deterministic retrieval metrics** (Recall@k / MRR) that do not depend on an
  LLM at all. The `RAG_TOPK` finding (count vs quality) points the same way:
  measuring retrieval quality needs ground-truth labels, not just counts.
- Still undecided (on purpose): whether this becomes an MVP-to-ship or stays
  an incremental experiment.
