"""Citation checks plus optional external factual-support annotations."""

from __future__ import annotations

from customer_service_rag.schemas import EvaluationExample, ResponseStatus, RunResult


def citation_scores(result: RunResult, example: EvaluationExample) -> dict:
    citations = set(result.response.citations)
    available = set()
    if result.retrieval:
        available = {passage.citation_id for passage in result.retrieval.passages}
    gold = set(example.gold_citations)
    if not citations:
        value = 0.0 if result.response.status == ResponseStatus.ANSWERED else None
        return {"evidence_validity": value, "gold_precision": value if gold else None}
    return {
        "evidence_validity": len(citations & available) / len(citations),
        "gold_precision": len(citations & gold) / len(citations) if gold else None,
    }


def factual_support_scores(annotation: dict | None) -> dict:
    if annotation is None:
        return {
            "hallucination_rate": None, "grounding_accuracy": None,
            "dialogue_consistency": None, "task_completed": None,
        }
    count = annotation.get("total_claims")
    unsupported = annotation.get("unsupported_claims")
    if count is not None and unsupported is not None:
        if not isinstance(count, int) or not isinstance(unsupported, int) or not 0 <= unsupported <= count:
            raise ValueError("Annotation claim counts must satisfy 0 <= unsupported <= total")
    rate = unsupported / count if count and unsupported is not None else None
    return {
        "hallucination_rate": rate,
        "grounding_accuracy": 1 - rate if rate is not None else None,
        "dialogue_consistency": annotation.get("dialogue_consistency"),
        "task_completed": annotation.get("task_completed"),
    }
