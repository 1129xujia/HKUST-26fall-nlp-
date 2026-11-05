"""Public contracts. Modules depend on these, not each other's implementation."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from customer_service_rag.schemas import (
    DialogueInput,
    EvaluationFeedback,
    GeneratedResponse,
    LLMResult,
    OptimizationAction,
    Passage,
    QueryPlan,
    RetrievalHit,
    RetrievalOptions,
    RetrievalResult,
    RunResult,
    RunState,
)


@runtime_checkable
class LLMClient(Protocol):
    async def complete(
        self, prompt: str, *, task: str, payload: dict[str, Any]
    ) -> LLMResult: ...


@runtime_checkable
class EmbeddingModel(Protocol):
    model_name: str
    dimensions: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


@runtime_checkable
class Planner(Protocol):
    async def plan(self, dialogue: DialogueInput) -> QueryPlan: ...


@runtime_checkable
class SearchBranch(Protocol):
    name: str

    def search(self, text: str, top_k: int, query_id: str) -> list[RetrievalHit]: ...


@runtime_checkable
class Retriever(Protocol):
    async def retrieve(
        self, query: QueryPlan, options: RetrievalOptions
    ) -> RetrievalResult: ...


@runtime_checkable
class Generator(Protocol):
    async def generate(
        self, query: QueryPlan, passages: list[Passage]
    ) -> GeneratedResponse: ...


@runtime_checkable
class Evaluator(Protocol):
    async def evaluate(
        self, query: QueryPlan, passages: list[Passage], response: GeneratedResponse
    ) -> EvaluationFeedback: ...


@runtime_checkable
class Optimizer(Protocol):
    async def optimize(
        self, state: RunState, feedback: EvaluationFeedback
    ) -> OptimizationAction: ...


@runtime_checkable
class Orchestrator(Protocol):
    async def run(self, dialogue: DialogueInput) -> RunResult: ...
