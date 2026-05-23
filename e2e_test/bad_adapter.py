"""A deliberately degraded adapter for testing regression detection."""


async def ask(query: str) -> dict:
    return {
        "answer": f"I don't know the answer to: {query}",
        "contexts": [
            {"id": "doc_99", "text": "Completely irrelevant document."},
        ],
    }
