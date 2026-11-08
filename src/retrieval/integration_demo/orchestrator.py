"""End-to-end wiring with only protocol-level coupling between parts."""

from dataclasses import dataclass

from retrieval.schemas import RetrievalRequest

from .contracts import QualityEvaluator, QueryPlanner, ResponseGenerator, RetrievalService
from .part1 import QueryPlan, RuleBasedPlanner
from .part3 import AnswerEvaluator, CitedAnswer, ExtractiveCitedGenerator


@dataclass(frozen=True)
class DemoResult:
    plan: QueryPlan
    evidence: list[dict]
    answer: CitedAnswer
    quality: dict
    retried_without_domain: bool

    def to_dict(self) -> dict:
        return {
            "plan": self.plan.to_dict(),
            "evidence": self.evidence,
            "answer": self.answer.to_dict(),
            "quality": self.quality,
            "retried_without_domain": self.retried_without_domain,
        }


class EndToEndDemo:
    def __init__(
        self,
        retrieval: RetrievalService,
        planner: QueryPlanner | None = None,
        generator: ResponseGenerator | None = None,
        evaluator: QualityEvaluator | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.planner = planner or RuleBasedPlanner()
        self.generator = generator or ExtractiveCitedGenerator()
        self.evaluator = evaluator or AnswerEvaluator()

    def run(
        self,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
        top_k: int = 5,
    ) -> DemoResult:
        plan = self.planner.plan(question, history)
        request = RetrievalRequest(
            query=plan.query,
            sub_queries=plan.sub_queries,
            domain=plan.domain,
            top_k=top_k,
        )
        evidence = self.retrieval.retrieve(request)
        retried = False
        if not evidence and plan.domain is not None:
            retried = True
            evidence = self.retrieval.retrieve(
                RetrievalRequest(
                    query=plan.query,
                    sub_queries=plan.sub_queries,
                    top_k=top_k,
                )
            )
        answer = self.generator.generate(question, evidence)
        quality = self.evaluator.evaluate(answer, evidence)
        return DemoResult(
            plan=plan,
            evidence=[item.to_dict() for item in evidence],
            answer=answer,
            quality=quality,
            retried_without_domain=retried,
        )
