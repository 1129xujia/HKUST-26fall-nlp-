from __future__ import annotations

from customer_service_rag.config import QualityConfig
from customer_service_rag.schemas import (
    Decision,
    EvaluationFeedback,
    GeneratedResponse,
    ResponseStatus,
)


class DecisionPolicy:
    def __init__(self, config: QualityConfig) -> None:
        self.config = config

    def apply(
        self, feedback: EvaluationFeedback, response: GeneratedResponse,
    ) -> EvaluationFeedback:
        if feedback.decision in {Decision.CLARIFY, Decision.ESCALATE}:
            return feedback
        failed = [
            name for name in (
                "relevance", "sufficiency", "groundedness", "dialogue_consistency",
            ) if getattr(feedback, name) < getattr(self.config, f"{name}_threshold")
        ]
        if response.status != ResponseStatus.ANSWERED or not response.citations:
            failed.append("supported_answer")
        if failed:
            return EvaluationFeedback.model_validate({
                **feedback.model_dump(mode="json"),
                "decision": "retry",
                "reason": feedback.reason + " Below policy requirements: " + ", ".join(failed),
                "suggested_action": feedback.suggested_action or "regenerate",
            })
        return feedback
