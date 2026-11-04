"""Deterministic Part 3 answer generator used to test evidence contracts."""

from dataclasses import asdict, dataclass
import re

from retrieval.schemas import Evidence


@dataclass(frozen=True)
class CitedAnswer:
    text: str
    citations: list[str]
    evidence_count: int

    def to_dict(self) -> dict:
        return asdict(self)


class ExtractiveCitedGenerator:
    def __init__(self, max_evidence: int = 3, max_chars_per_evidence: int = 260) -> None:
        self.max_evidence = max_evidence
        self.max_chars_per_evidence = max_chars_per_evidence

    def generate(self, question: str, evidence: list[Evidence]) -> CitedAnswer:
        if not evidence:
            return CitedAnswer(
                text="I could not find enough grounded evidence to answer this question.",
                citations=[],
                evidence_count=0,
            )
        statements: list[str] = []
        citations: list[str] = []
        for item in evidence[: self.max_evidence]:
            sentence = self._first_sentence(item.text)[: self.max_chars_per_evidence].strip()
            if not sentence:
                continue
            marker = f"[{item.passage_id}]"
            statements.append(f"{sentence} {marker}")
            citations.append(item.passage_id)
        prefix = f"For “{question}”, the retrieved guidance says: "
        return CitedAnswer(
            text=prefix + " ".join(statements),
            citations=citations,
            evidence_count=len(evidence),
        )

    @staticmethod
    def _first_sentence(text: str) -> str:
        compact = " ".join(text.split())
        match = re.search(r".+?(?:[.!?](?:\s|$)|$)", compact)
        return match.group(0).strip() if match else compact
