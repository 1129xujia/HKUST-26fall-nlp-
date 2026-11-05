"""Normalize and bound model-generated retrieval sub-questions."""

from customer_service_rag.planning.rewriting import normalize_query


def normalize_sub_queries(values: list[str], rewritten: str, limit: int) -> list[str]:
    seen = {rewritten.casefold()}
    result = []
    for value in values:
        text = normalize_query(value)
        if text.casefold() not in seen:
            seen.add(text.casefold())
            result.append(text)
    return result[:limit]
