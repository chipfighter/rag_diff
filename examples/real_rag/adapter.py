"""rag-diff adapter for the example RAG pipeline.

Exposes ``ask(query) -> {answer, contexts}``. Two environment knobs let you
deliberately introduce regressions so rag-diff's triage can be exercised:

    RAG_TOPK         number of chunks to retrieve (default 4)
    RAG_RETRIEVAL    "good" (default) or "bad" (return the least-relevant chunks)
    RAG_PROMPT_MODE  "grounded" (default) or "lazy" (degrade generation)
"""

import os
import sys
from pathlib import Path

# rag-diff loads this file by path, so its own directory is not yet importable.
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import generate, get_index  # noqa: E402


async def ask(query: str) -> dict:
    top_k = int(os.environ.get("RAG_TOPK") or 4)
    prompt_mode = os.environ.get("RAG_PROMPT_MODE") or "grounded"
    worst = (os.environ.get("RAG_RETRIEVAL") or "good") == "bad"

    hits = get_index().retrieve(query, top_k=top_k, worst=worst)
    contexts = [
        {"id": chunk.id, "text": chunk.text, "score": round(score, 4)}
        for chunk, score in hits
    ]
    answer = await generate(query, [c["text"] for c in contexts], prompt_mode)
    return {"answer": answer, "contexts": contexts}
