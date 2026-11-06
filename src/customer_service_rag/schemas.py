"""Version 1 data contracts shared by all project modules."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonEmpty = Annotated[str, Field(min_length=1)]
Score = Annotated[float, Field(ge=0.0, le=1.0)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ChatMessage(Contract):
    role: Literal["user", "agent", "system"]
    text: NonEmpty


class DialogueInput(Contract):
    conversation_id: NonEmpty
    current_message: NonEmpty
    dialogue_history: list[ChatMessage] = Field(default_factory=list)


class QueryPlan(DialogueInput):
    rewritten_query: NonEmpty
    intent: str = "unknown"
    domain: str = "unknown"
    sub_queries: list[NonEmpty] = Field(default_factory=list)

    def retrieval_queries(self) -> list[tuple[str, str]]:
        values = [("original", self.current_message), ("rewritten", self.rewritten_query)]
        values.extend((f"sub_{i}", text) for i, text in enumerate(self.sub_queries, 1))
        seen: set[str] = set()
        result = []
        for query_id, text in values:
            key = " ".join(text.casefold().split())
            if key not in seen:
                seen.add(key)
                result.append((query_id, text))
        return result


class Passage(Contract):
    doc_id: NonEmpty
    passage_id: NonEmpty
    title: NonEmpty
    text: NonEmpty
    domain: str = "unknown"
    span_ids: list[str] = Field(default_factory=list)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    source: str = "synthetic"
    original_doc_id: str | None = None

    @model_validator(mode="after")
    def validate_offsets(self) -> Passage:
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("char_start and char_end must be supplied together")
        if self.char_start is not None and self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        if ":" in self.doc_id or ":" in self.passage_id:
            raise ValueError("IDs must not contain the citation separator ':'")
        return self

    @property
    def citation_id(self) -> str:
        return f"{self.doc_id}:{self.passage_id}"


class RetrievalHit(Contract):
    passage: Passage
    source: Literal["dense", "bm25"]
    query_id: NonEmpty
    score: float = Field(allow_inf_nan=False)
    rank: int = Field(ge=1)


class RetrievalCandidate(Contract):
    passage: Passage
    hits: list[RetrievalHit]
    fusion_score: float | None = Field(default=None, allow_inf_nan=False)
    rerank_score: float | None = Field(default=None, allow_inf_nan=False)


class RetrievalOptions(Contract):
    branches: list[Literal["dense", "bm25"]] = Field(
        default_factory=lambda: ["dense", "bm25"], min_length=1
    )
    per_branch_top_k: int = Field(default=10, ge=1)
    fusion: Literal["rrf", "interleave"] = "rrf"
    rrf_k: int = Field(default=60, ge=1)
    candidate_top_k: int = Field(default=10, ge=1)
    rerank: bool = False
    final_top_k: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def validate_limits(self) -> RetrievalOptions:
        if len(self.branches) != len(set(self.branches)):
            raise ValueError("retrieval branches must be unique")
        if self.final_top_k > self.candidate_top_k:
            raise ValueError("final_top_k must not exceed candidate_top_k")
        return self


class RetrievalResult(Contract):
    passages: list[Passage] = Field(default_factory=list)
    candidates: list[RetrievalCandidate] = Field(default_factory=list)
    hits: list[RetrievalHit] = Field(default_factory=list)
    elapsed_ms: float = Field(default=0.0, ge=0)
    branch_elapsed_ms: dict[str, float] = Field(default_factory=dict)
    embedding_backend: str = "hash"
    rerank_elapsed_ms: float = Field(default=0.0, ge=0)
    rerank_model: str | None = None


class ResponseStatus(str, Enum):
    ANSWERED = "answered"
    CLARIFICATION_NEEDED = "clarification_needed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ESCALATED = "escalated"
    ERROR = "error"


class GeneratedResponse(Contract):
    answer: NonEmpty
    citations: list[str] = Field(default_factory=list)
    retrieval_round: int = Field(default=0, ge=0)
    status: ResponseStatus = ResponseStatus.ANSWERED


class Decision(str, Enum):
    PASS = "pass"
    RETRY = "retry"
    CLARIFY = "clarify"
    ESCALATE = "escalate"


class EvaluationFeedback(Contract):
    relevance: Score
    sufficiency: Score
    groundedness: Score
    dialogue_consistency: Score
    decision: Decision
    missing_information: str = ""
    suggested_action: str = ""
    reason: str = ""
    clarification_question: NonEmpty | None = None
    suggested_queries: list[NonEmpty] = Field(default_factory=list)


class OptimizationAction(Contract):
    action: Literal[
        "rewrite_query", "add_sub_queries", "adjust_retrieval",
        "regenerate", "clarify", "escalate",
    ]
    reason: NonEmpty
    rewritten_query: NonEmpty | None = None
    sub_queries: list[NonEmpty] = Field(default_factory=list)
    retrieval_options: RetrievalOptions | None = None
    clarification_question: NonEmpty | None = None

    @model_validator(mode="after")
    def validate_action(self) -> OptimizationAction:
        fields = {
            "rewritten_query": self.rewritten_query,
            "sub_queries": self.sub_queries,
            "retrieval_options": self.retrieval_options,
            "clarification_question": self.clarification_question,
        }
        allowed = {
            "rewrite_query": {"rewritten_query"},
            "add_sub_queries": {"sub_queries"},
            "adjust_retrieval": {"retrieval_options"},
            "regenerate": set(),
            "clarify": {"clarification_question"},
            "escalate": set(),
        }[self.action]
        for name, value in fields.items():
            supplied = value is not None and value != []
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{name} must not be blank")
            if isinstance(value, list) and any(not text.strip() for text in value):
                raise ValueError(f"{name} must not contain blank strings")
            if name in allowed and not supplied:
                raise ValueError(f"{self.action} requires {name}")
            if name not in allowed and supplied:
                raise ValueError(f"{self.action} does not accept {name}")
        return self


class TokenUsage(Contract):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    source: Literal["estimated", "reported", "counted", "unavailable"] = "unavailable"

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


class LLMResult(Contract):
    text: str
    model: str
    is_mock: bool
    usage: TokenUsage = Field(default_factory=TokenUsage)


class RunPhase(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    RETRIEVING = "retrieving"
    GENERATING = "generating"
    EVALUATING = "evaluating"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    FAILED = "failed"


class TraceEvent(Contract):
    event: str = "transition"
    phase: RunPhase
    iteration: int = Field(default=0, ge=0)
    retrieval_round: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    elapsed_ms: float = Field(ge=0)
    details: dict[str, Any] = Field(default_factory=dict)


class RunState(Contract):
    run_id: NonEmpty
    dialogue: DialogueInput
    phase: RunPhase = RunPhase.CREATED
    query: QueryPlan | None = None
    retrieval: RetrievalResult | None = None
    response: GeneratedResponse | None = None
    feedback: list[EvaluationFeedback] = Field(default_factory=list)
    actions: list[OptimizationAction] = Field(default_factory=list)
    retrieval_options: RetrievalOptions | None = None
    reranker_available: bool = False
    retrieval_history: list[RetrievalResult] = Field(default_factory=list)
    candidate_history: list[GeneratedResponse] = Field(default_factory=list)
    iteration: int = Field(default=0, ge=0)
    retrieval_round: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    seen_evidence_ids: set[str] = Field(default_factory=set)
    usage: list[TokenUsage] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    stop_reason: str | None = None


class EscalationPackage(Contract):
    reason: NonEmpty
    conversation_id: NonEmpty
    current_message: NonEmpty
    dialogue_history: list[ChatMessage]
    evidence_ids: list[str] = Field(default_factory=list)
    evidence: list[Passage] = Field(default_factory=list)
    query: QueryPlan | None = None
    attempted_actions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    evaluation_feedback: list[EvaluationFeedback] = Field(default_factory=list)
    delivered_to_human: bool = False


class RunResult(Contract):
    run_id: NonEmpty
    sample_id: str | None = None
    conversation_id: NonEmpty
    response: GeneratedResponse
    query: QueryPlan | None = None
    retrieval: RetrievalResult | None = None
    feedback: list[EvaluationFeedback] = Field(default_factory=list)
    actions: list[OptimizationAction] = Field(default_factory=list)
    retrieval_options: RetrievalOptions | None = None
    retrieval_history: list[RetrievalResult] = Field(default_factory=list)
    candidate_history: list[GeneratedResponse] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    usage: list[TokenUsage] = Field(default_factory=list)
    elapsed_ms: float = Field(ge=0)
    iteration: int = Field(ge=0)
    retrieval_round: int = Field(ge=0)
    retry_count: int = Field(default=0, ge=0)
    stop_reason: NonEmpty
    is_mock: bool = True
    escalation: EscalationPackage | None = None
    mock_scenario: str | None = None
    model_backend: str = "mock"
    model_calls: list[dict[str, Any]] = Field(default_factory=list)
    pipeline_backend: str = "legacy"
    integration: dict[str, Any] = Field(default_factory=dict)


class GroundingReference(Contract):
    doc_id: NonEmpty
    passage_id: str | None = None
    span_ids: list[str] = Field(default_factory=list)
    original_doc_id: str | None = None
    label: str | None = None
    char_start: int | None = None
    char_end: int | None = None


class EvaluationExample(Contract):
    sample_id: NonEmpty
    split: Literal["train", "dev", "validation", "test", "mini"]
    dialogue: DialogueInput
    reference_answers: list[str] = Field(default_factory=list)
    gold_citations: list[str] = Field(default_factory=list)
    gold_evidence_groups: list[Annotated[list[NonEmpty], Field(min_length=1)]] = Field(
        default_factory=list
    )
    grounding: list[GroundingReference] = Field(default_factory=list)
    expected_behavior: Literal["answer", "clarify", "escalate"] | None = None
    source: str = "synthetic"
    domain: str | None = None
    origin_split: str | None = None
    user_turn_ids: list[int] = Field(default_factory=list)
