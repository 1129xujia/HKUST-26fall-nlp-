"""Async component boundaries. Injected model clients must charge the ledger."""

from typing import Protocol

from customer_service_rag.schemas import GeneratedResponse, QueryPlan

from .budget import BudgetLedger
from .schemas import (
    EvaluationRequest, ExecutionOptions, Feedback, GenerationRequest,
    QueryUpdate, RetrievalSnapshot, RewriteRequest,
)


class RevisionGenerator(Protocol):
    async def generate(self, request: GenerationRequest,
                       budget: BudgetLedger) -> GeneratedResponse: ...


class QueryRewriter(Protocol):
    async def rewrite(self, request: RewriteRequest, budget: BudgetLedger) -> QueryUpdate: ...


class QualityEvaluator(Protocol):
    async def evaluate(self, request: EvaluationRequest, budget: BudgetLedger) -> Feedback: ...


class RetrievalExecutor(Protocol):
    async def retrieve(self, query: QueryPlan, options: ExecutionOptions,
                       snapshot: RetrievalSnapshot | None = None,
                       trace_id: str | None = None) -> RetrievalSnapshot: ...
