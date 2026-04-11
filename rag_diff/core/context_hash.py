"""Context hashing utilities for tracking retrieval changes.

Uses SHA-256 to generate deterministic fingerprints of context items,
enabling diff detection between runs even when contexts are plain text
without explicit IDs.
"""

import hashlib
import re
from typing import Union


def normalize_text(text: str) -> str:
    """Normalize text by stripping edges and collapsing internal whitespace."""
    return re.sub(r"\s+", " ", text.strip())


def hash_context(context: Union[str, dict]) -> str:
    """Compute a SHA-256 hash of a context item.

    Accepts plain text strings or structured dicts. For dicts, uses the
    'text' field if present, falling back to 'id', then empty string.
    """
    if isinstance(context, str):
        text = context
    elif isinstance(context, dict):
        text = context.get("text", context.get("id", ""))
    else:
        text = str(context)

    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def hash_context_list(contexts: list[Union[str, dict]]) -> list[str]:
    """Compute sorted SHA-256 hashes for a list of context items.

    Sorting ensures that the hash list is order-independent, so two runs
    returning the same contexts in different order are treated as identical.
    """
    return sorted(hash_context(ctx) for ctx in contexts)
