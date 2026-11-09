"""Small protocols shared by the application and model adapters."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from customer_service_rag.schemas import DialogueInput, LLMResult, QueryPlan, RunResult


@runtime_checkable
class LLMClient(Protocol):
    async def complete(self, prompt: str, *, task: str, payload: dict[str, Any]) -> LLMResult: ...


@runtime_checkable
class EmbeddingModel(Protocol):
    model_name: str
    dimensions: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


@runtime_checkable
class Planner(Protocol):
    async def plan(self, dialogue: DialogueInput) -> QueryPlan: ...


@runtime_checkable
class Orchestrator(Protocol):
    async def run(self, dialogue: DialogueInput) -> RunResult: ...
