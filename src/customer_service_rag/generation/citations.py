from __future__ import annotations

from customer_service_rag.schemas import GeneratedResponse, Passage, ResponseStatus


def normalize_citations(values, aliases: dict[str, str]) -> list[str]:
    """Resolve only explicitly supplied IDs; never infer an ID from a title or text."""
    if isinstance(values, (str, dict)):
        values = [values]
    if not isinstance(values, list):
        raise ValueError("citations must be an array of IDs")
    result = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("id")
        if not isinstance(value, str):
            raise ValueError("Each citation requires a string evidence ID")
        key = value.strip().strip("[]")
        citation = aliases.get(key.upper(), value)
        if citation not in result:
            result.append(citation)
    return result


def validate_citations(response: GeneratedResponse, passages: list[Passage]) -> None:
    valid = {passage.citation_id for passage in passages}
    if not set(response.citations).issubset(valid):
        raise ValueError("Response contains citations outside the supplied evidence")
    if response.status == ResponseStatus.ANSWERED and not response.citations:
        raise ValueError("An answered response must contain at least one citation")
