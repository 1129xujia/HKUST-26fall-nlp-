"""Rank fusion algorithms."""

from collections import defaultdict

from .schemas import FusedHit, SearchHit


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchHit]],
    *,
    rrf_k: int = 60,
) -> list[FusedHit]:
    """Fuse ranked lists by passage ID using Reciprocal Rank Fusion."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    scores: dict[str, float] = defaultdict(float)
    fused: dict[str, FusedHit] = {}

    for results in ranked_lists:
        seen_in_list: set[str] = set()
        for fallback_rank, hit in enumerate(results, start=1):
            rank = hit.rank if hit.rank > 0 else fallback_rank
            passage_id = hit.passage.passage_id
            if passage_id in seen_in_list:
                continue
            seen_in_list.add(passage_id)
            scores[passage_id] += 1.0 / (rrf_k + rank)

            item = fused.setdefault(passage_id, FusedHit(passage=hit.passage))
            if hit.retriever not in item.source_retrievers:
                item.source_retrievers.append(hit.retriever)
            if hit.matched_query not in item.matched_queries:
                item.matched_queries.append(hit.matched_query)

            if hit.retriever == "dense":
                item.dense_rank = min(
                    rank,
                    item.dense_rank if item.dense_rank is not None else rank,
                )
            elif hit.retriever == "bm25":
                item.bm25_rank = min(
                    rank,
                    item.bm25_rank if item.bm25_rank is not None else rank,
                )

    for passage_id, score in scores.items():
        fused[passage_id].rrf_score = score
        fused[passage_id].source_retrievers.sort()

    return sorted(
        fused.values(),
        key=lambda item: (-item.rrf_score, item.passage.passage_id),
    )
