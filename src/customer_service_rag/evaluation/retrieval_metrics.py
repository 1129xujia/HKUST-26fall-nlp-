"""Evidence-group recall with alternative overlapping passages."""

from __future__ import annotations

from customer_service_rag.schemas import EvaluationExample, RetrievalResult


def evidence_scores(ranking: list[str], groups: list[list[str]], k: int) -> dict:
    if not groups:
        return {"recall": None, "hit": None, "complete_coverage": None}
    selected = set(ranking[:k])
    hits = sum(bool(selected.intersection(group)) for group in groups)
    return {
        "recall": hits / len(groups),
        "hit": float(hits > 0),
        "complete_coverage": float(hits == len(groups)),
    }


def retrieval_scores(result: RetrievalResult | None, example: EvaluationExample) -> dict:
    groups = example.gold_evidence_groups or [[key] for key in example.gold_citations]
    ranking = []
    if result:
        ranking = [candidate.passage.citation_id for candidate in result.candidates]
    return {
        f"@{k}": evidence_scores(ranking, groups, k) for k in [1, 5, 10]
    }
