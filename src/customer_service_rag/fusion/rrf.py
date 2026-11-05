from __future__ import annotations

from customer_service_rag.fusion.deduplication import group_hits
from customer_service_rag.schemas import RetrievalCandidate, RetrievalHit


def reciprocal_rank_fusion(
    rankings: list[list[RetrievalHit]], k: int = 60,
) -> list[RetrievalCandidate]:
    if k < 1:
        raise ValueError("RRF k must be positive")
    candidates = group_hits(rankings)
    for candidate in candidates.values():
        candidate.fusion_score = sum(1.0 / (k + hit.rank) for hit in candidate.hits)
    return sorted(
        candidates.values(),
        key=lambda item: (-item.fusion_score, item.passage.citation_id),
    )
