"""Sample RAG adapter for rag-diff.

Replace this with your actual RAG pipeline. The function must accept
a query string and return a dict with 'answer' and 'contexts' keys.
"""


async def ask(query: str) -> dict:
    """A dummy adapter that echoes the query."""
    return {
        "answer": f"This is a sample answer to: {query}",
        "contexts": [
            {"id": "doc_1", "text": "Sample context document one."},
            {"id": "doc_2", "text": "Sample context document two."},
        ],
    }
