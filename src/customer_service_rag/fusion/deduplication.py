from __future__ import annotations

from customer_service_rag.schemas import RetrievalCandidate, RetrievalHit


def group_hits(rankings: list[list[RetrievalHit]]) -> dict[str, RetrievalCandidate]:
    candidates: dict[str, RetrievalCandidate] = {}
    for ranking in rankings:
        for hit in ranking:
            key = hit.passage.citation_id
            if key not in candidates:
                candidates[key] = RetrievalCandidate(passage=hit.passage, hits=[])
            candidates[key].hits.append(hit)
    return candidates


def interleave(rankings: list[list[RetrievalHit]]) -> list[RetrievalCandidate]:
    candidates = group_hits(rankings)
    seen: set[str] = set()
    result = []
    for offset in range(max((len(items) for items in rankings), default=0)):
        for ranking in rankings:
            if offset < len(ranking):
                key = ranking[offset].passage.citation_id
                if key not in seen:
                    seen.add(key)
                    result.append(candidates[key])
    return result
