"""A minimal but real RAG pipeline used as the subject-under-test for rag-diff.

corpus (markdown) -> chunk -> local embeddings (sentence-transformers)
-> NumPy cosine-similarity retrieval -> DeepSeek answer.

Everything here is deliberately small and readable: this is a proof-of-concept
RAG system, not a production retriever.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

CORPUS_DIR = Path(__file__).parent / "corpus"
EMBED_MODEL = "all-MiniLM-L6-v2"
GEN_MODEL = os.environ.get("RAG_GEN_MODEL", "deepseek-chat")


@dataclass
class Chunk:
    id: str
    text: str


def load_chunks(corpus_dir: Path = CORPUS_DIR) -> list[Chunk]:
    """Read every .md file and split it into paragraph-sized chunks."""
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        paragraphs = [p.strip() for p in path.read_text(encoding="utf-8").split("\n\n")]
        for i, para in enumerate(p for p in paragraphs if p):
            chunks.append(Chunk(id=f"{path.stem}#{i}", text=para))
    return chunks


class RagIndex:
    """Builds embeddings for the corpus once and answers top-k queries."""

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, model) -> None:
        self.chunks = chunks
        self.embeddings = embeddings  # shape (N, D), L2-normalized
        self._model = model

    @classmethod
    def build(cls) -> "RagIndex":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBED_MODEL)
        chunks = load_chunks()
        vectors = model.encode([c.text for c in chunks], normalize_embeddings=True)
        return cls(chunks, np.asarray(vectors, dtype=np.float32), model)

    def retrieve(
        self, query: str, top_k: int, worst: bool = False
    ) -> list[tuple[Chunk, float]]:
        q = self._model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ np.asarray(q, dtype=np.float32)  # cosine (normalized)
        order = np.argsort(scores)  # ascending: least similar first
        if not worst:
            order = order[::-1]  # most similar first
        order = order[: max(1, top_k)]
        return [(self.chunks[i], float(scores[i])) for i in order]


@lru_cache(maxsize=1)
def get_index() -> RagIndex:
    """Build the index once and reuse it across all calls in the process."""
    return RagIndex.build()


PROMPTS = {
    "grounded": (
        "Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer:"
    ),
    # Deliberately degraded: ignores the retrieved context and answers from the
    # model's own memory in one short line. Used to simulate a "model regression"
    # while retrieval stays identical.
    "lazy": (
        "Answer the question in one short sentence from your own general "
        "knowledge. Do not use or read any provided documents.\n\n"
        "Question: {query}\n\n"
        "Answer:"
    ),
}


async def generate(query: str, contexts: list[str], prompt_mode: str = "grounded") -> str:
    """Generate an answer with DeepSeek via the OpenAI-compatible API."""
    from openai import AsyncOpenAI

    template = PROMPTS.get(prompt_mode, PROMPTS["grounded"])
    prompt = template.format(context="\n\n".join(contexts), query=query)

    client = AsyncOpenAI()  # reads OPENAI_API_KEY + OPENAI_BASE_URL from env (.env)
    resp = await client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


if __name__ == "__main__":
    # Retrieval-only sanity check (no API key needed).
    index = get_index()
    print(f"Loaded {len(index.chunks)} chunks from {CORPUS_DIR}")
    for q in ["weekend overtime pay", "annual leave after 5 years"]:
        print(f"\nQuery: {q!r}")
        for chunk, score in index.retrieve(q, top_k=3):
            print(f"  {score:.3f}  {chunk.id}: {chunk.text[:60]}...")
