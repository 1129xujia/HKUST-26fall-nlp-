"""Contract-level checks for grounded demo answers."""

import re

from retrieval.schemas import Evidence

from .generator import CitedAnswer


class AnswerEvaluator:
    def evaluate(self, answer: CitedAnswer, evidence: list[Evidence]) -> dict:
        available = {item.passage_id for item in evidence}
        cited = set(answer.citations)
        inline = set(re.findall(r"\[([^\[\]]+)\]", answer.text))
        valid = cited <= available and cited <= inline
        citation_coverage = 0.0 if not evidence else len(cited & available) / len(evidence)
        return {
            "has_evidence": bool(evidence),
            "has_answer": bool(answer.text.strip()),
            "citations_valid": valid,
            "citation_coverage": citation_coverage,
            "grounded": bool(evidence) and valid and bool(cited),
        }
