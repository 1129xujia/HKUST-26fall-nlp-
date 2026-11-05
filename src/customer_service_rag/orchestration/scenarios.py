"""Explicit scripted mock services for CLI demonstrations, never dataset gold labels."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from customer_service_rag.config import read_yaml
from customer_service_rag.generation.generator import ResponseGenerator
from customer_service_rag.interfaces import LLMClient
from customer_service_rag.orchestration.loop import (
    EvaluatorFactory,
    GeneratorFactory,
    OptimizerFactory,
)
from customer_service_rag.schemas import (
    Contract,
    DialogueInput,
    EvaluationFeedback,
    GeneratedResponse,
    OptimizationAction,
    Passage,
    QueryPlan,
    ResponseStatus,
    RunState,
)


class MockScenario(Contract):
    name: str
    description: str
    dialogue: DialogueInput
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    feedback: list[EvaluationFeedback] = Field(default_factory=list, min_length=1)
    actions: list[OptimizationAction] = Field(default_factory=list)
    generation_modes: list[Literal["extract", "incomplete"]] = Field(default_factory=list)
    evaluation_delay_seconds: float = Field(default=0.0, ge=0, le=5)


def load_scenarios(path: Path) -> dict[str, MockScenario]:
    data = read_yaml(path)
    if data.get("scenario_version") != 1 or not isinstance(data.get("scenarios"), dict):
        raise ValueError("Expected scenario_version: 1 and a scenarios mapping")
    if not data["scenarios"]:
        raise ValueError("Scenario catalog is empty")
    return {
        name: MockScenario.model_validate({"name": name, **values})
        for name, values in data["scenarios"].items()
    }


class ScriptedEvaluator:
    def __init__(self, scenario: MockScenario) -> None:
        self.scenario = scenario
        self.position = 0

    async def evaluate(
        self, query: QueryPlan, passages: list[Passage], response: GeneratedResponse,
    ) -> EvaluationFeedback:
        if self.scenario.evaluation_delay_seconds:
            await asyncio.sleep(self.scenario.evaluation_delay_seconds)
        if self.position >= len(self.scenario.feedback):
            raise ValueError(f"Mock feedback exhausted in scenario {self.scenario.name}")
        result = self.scenario.feedback[self.position].model_copy(deep=True)
        self.position += 1
        return result


class ScriptedOptimizer:
    def __init__(self, scenario: MockScenario) -> None:
        self.scenario = scenario
        self.position = 0

    async def optimize(
        self, state: RunState, feedback: EvaluationFeedback,
    ) -> OptimizationAction:
        if self.position >= len(self.scenario.actions):
            raise ValueError(f"Mock actions exhausted in scenario {self.scenario.name}")
        result = self.scenario.actions[self.position].model_copy(deep=True)
        self.position += 1
        return result


class ScriptedGenerator:
    def __init__(self, scenario: MockScenario, llm: LLMClient) -> None:
        self.scenario = scenario
        self.fallback = ResponseGenerator(llm)
        self.position = 0

    async def generate(self, query: QueryPlan, passages: list[Passage]) -> GeneratedResponse:
        modes = self.scenario.generation_modes
        mode = modes[self.position] if self.position < len(modes) else "extract"
        self.position += 1
        if mode == "incomplete":
            return GeneratedResponse(
                answer="The mock draft is incomplete and does not yet provide a supported answer.",
                status=ResponseStatus.INSUFFICIENT_EVIDENCE,
            )
        return await self.fallback.generate(query, passages)


def scenario_factories(
    scenario: MockScenario,
) -> tuple[EvaluatorFactory, OptimizerFactory, GeneratorFactory]:
    return (
        lambda llm: ScriptedEvaluator(scenario),
        lambda llm: ScriptedOptimizer(scenario),
        lambda llm: ScriptedGenerator(scenario, llm),
    )
