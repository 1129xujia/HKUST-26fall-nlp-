"""Public, JSON-serializable contracts for the feedback-driven optimizer.

The package name intentionally follows the requested ``optimaization_v2`` path.
Part 2's existing Evidence contract and main's dialogue/answer contracts are reused.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from customer_service_rag.schemas import (
    EvaluationFeedback, GeneratedResponse, QueryPlan,
)
from retrieval_v2.retrieval.schemas import Evidence, FusedHit, SearchHit

FailureType = Literal[
    "answer_defect", "query_misunderstanding", "retrieval_coverage",
    "retrieval_balance", "evidence_ranking", "missing_user_information",
]
Branch = Literal[
    "pass", "regenerate", "rewrite_query", "adjust_hybrid", "rerank",
    "clarify", "escalate", "stop",
]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class Feedback(EvaluationFeedback):
    """Accepts main EvaluationFeedback without requiring new upstream fields."""

    failure_types: list[FailureType] = Field(default_factory=list)
    evidence_sufficient: bool | None = None
    preferred_retriever: Literal["dense", "bm25"] | None = None

    @property
    def needs_optimization(self) -> bool:
        return self.decision.value == "retry"


class ExecutionOptions(Contract):
    """Request-local settings, separate from Part 2 RetrievalRequest."""

    dense_top_k: int = Field(default=100, ge=0, le=500, strict=True)
    bm25_top_k: int = Field(default=50, ge=0, le=500, strict=True)
    dense_weight: float = Field(default=1.0, ge=0, le=10)
    bm25_weight: float = Field(default=1.0, ge=0, le=10)
    rrf_k: int = Field(default=30, ge=1, le=200, strict=True)
    candidate_top_k: int = Field(default=50, ge=1, le=500, strict=True)
    final_top_k: int = Field(default=6, ge=1, le=20, strict=True)
    rerank: bool = False
    reranker_query_strategy: Literal["identity", "current_first_user_history"] = (
        "current_first_user_history"
    )
    reranker_use_sub_queries: bool = False
    reranker_skip_acknowledgements: bool = True
    # A planner domain is only a hint unless the caller explicitly trusts it.
    domain_filter: bool = False

    @model_validator(mode="after")
    def validate_branches(self):
        if not (
            (self.dense_top_k > 0 and self.dense_weight > 0)
            or (self.bm25_top_k > 0 and self.bm25_weight > 0)
        ):
            raise ValueError("At least one retrieval branch must have quota and weight")
        if self.candidate_top_k < self.final_top_k:
            raise ValueError("candidate_top_k must cover final_top_k")
        return self


class Capabilities(Contract):
    generation: bool = True
    rewriting: bool = True
    hybrid: bool = True
    reranking: bool = False
    retrievers: list[Literal["dense", "bm25"]] = Field(
        default_factory=lambda: ["dense", "bm25"]
    )


class Limits(Contract):
    max_retries: int = Field(default=2, ge=0, le=10, strict=True)
    max_total_tokens: int = Field(default=20000, ge=1, strict=True)
    max_latency_seconds: float = Field(default=180, gt=0)
    escalation_enabled: bool = True
    relevance_threshold: float = Field(default=0.7, ge=0, le=1)
    sufficiency_threshold: float = Field(default=0.7, ge=0, le=1)
    groundedness_threshold: float = Field(default=0.8, ge=0, le=1)
    dialogue_consistency_threshold: float = Field(default=0.8, ge=0, le=1)


class BudgetState(Contract):
    # The caller must include upstream costs to enforce an end-to-end budget.
    used_tokens: int | None = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0, ge=0)


class RetrievalSnapshot(Contract):
    recall_key: str
    corpus_version: str
    rankings: list[list[SearchHit]] = Field(default_factory=list)
    candidates: list[FusedHit] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    reused_recall: bool = False
    branch_counts: dict[str, int] = Field(default_factory=dict)
    elapsed_ms: float = Field(default=0, ge=0)
    notes: list[str] = Field(default_factory=list)


class Revision(Contract):
    instruction: str = Field(min_length=1)
    missing_information: str = ""
    evaluation_reason: str = ""


class OptimizationAction(Contract):
    branch: Branch
    reason: str = Field(min_length=1)
    revision: Revision | None = None
    options: ExecutionOptions | None = None
    clarification_question: str | None = None

    @model_validator(mode="after")
    def validate_payload(self):
        if (self.branch == "regenerate") != (self.revision is not None):
            raise ValueError("Only regenerate requires a revision")
        if (self.branch in {"adjust_hybrid", "rerank"}) != (self.options is not None):
            raise ValueError("Only adjust_hybrid/rerank require options")
        if self.branch == "clarify":
            if not self.clarification_question or not self.clarification_question.strip():
                raise ValueError("clarify requires a question")
        elif self.clarification_question is not None:
            raise ValueError("Only clarify accepts clarification_question")
        return self


class OptimizationInput(Contract):
    query: QueryPlan
    response: GeneratedResponse
    evidence: list[Evidence]
    feedback: Feedback
    options: ExecutionOptions = Field(default_factory=ExecutionOptions)
    snapshot: RetrievalSnapshot | None = None
    capabilities: Capabilities = Field(default_factory=Capabilities)
    limits: Limits = Field(default_factory=Limits)
    budget: BudgetState = Field(default_factory=BudgetState)
    retry_count: int = Field(default=0, ge=0)
    action_history: list[OptimizationAction] = Field(default_factory=list)
    trace_id: str | None = None

    @model_validator(mode="after")
    def unique_evidence(self):
        ids = [item.passage_id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("Evidence passage IDs must be unique")
        return self


class GenerationRequest(Contract):
    query: QueryPlan
    evidence: list[Evidence]
    previous_response: GeneratedResponse
    feedback: Feedback
    revision: Revision | None = None


class RewriteRequest(GenerationRequest):
    options: ExecutionOptions
    action_history: list[OptimizationAction] = Field(default_factory=list)


class QueryUpdate(Contract):
    rewritten_query: str = Field(min_length=1)
    sub_queries: list[str] = Field(default_factory=list)

    @field_validator("rewritten_query")
    @classmethod
    def clean_query(cls, value):
        value = " ".join(value.split())
        if not value:
            raise ValueError("Query cannot be blank")
        return value

    @model_validator(mode="after")
    def clean_sub_queries(self):
        seen = {self.rewritten_query.casefold()}
        cleaned = []
        for text in self.sub_queries:
            text = " ".join(text.split())
            if text and text.casefold() not in seen:
                cleaned.append(text)
                seen.add(text.casefold())
        # object.__setattr__ avoids recursive validate_assignment here.
        object.__setattr__(self, "sub_queries", cleaned[:4])
        return self


class EvaluationRequest(Contract):
    query: QueryPlan
    evidence: list[Evidence]
    response: GeneratedResponse


class TraceEntry(Contract):
    event: str
    details: dict = Field(default_factory=dict)


class OptimizationResult(Contract):
    response: GeneratedResponse
    query: QueryPlan
    evidence: list[Evidence]
    options: ExecutionOptions
    feedback_history: list[Feedback]
    candidate_history: list[GeneratedResponse]
    actions: list[OptimizationAction]
    trace: list[TraceEntry]
    retry_count: int
    retrieval_rounds: int
    evaluations: int
    used_tokens: int | None
    elapsed_seconds: float
    stop_reason: str
    snapshot: RetrievalSnapshot | None = None
    escalation: dict | None = None
