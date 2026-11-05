"""Normalization of a model-produced standalone retrieval query."""


def normalize_query(text: str) -> str:
    text = " ".join(text.split())
    if not text:
        raise ValueError("The rewritten query is empty")
    return text
