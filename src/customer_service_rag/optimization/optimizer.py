from __future__ import annotations

import json
from importlib.resources import files

from jinja2 import Environment, StrictUndefined

from customer_service_rag.interfaces import LLMClient
from customer_service_rag.schemas import EvaluationFeedback, OptimizationAction, RunState


class OptimizerAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def optimize(
        self, state: RunState, feedback: EvaluationFeedback,
    ) -> OptimizationAction:
        if state.query is None or state.retrieval_options is None:
            raise ValueError("Optimization requires a query and current retrieval options")
        payload = {
            "query": state.query.model_dump(mode="json"),
            "retrieval_options": state.retrieval_options.model_dump(mode="json"),
            "feedback": feedback.model_dump(mode="json"),
            "retry_count": state.retry_count,
        }
        template = files("customer_service_rag").joinpath("prompts/optimization.jinja2")
        prompt = Environment(undefined=StrictUndefined).from_string(
            template.read_text(encoding="utf-8")
        ).render(input_json=json.dumps(payload, ensure_ascii=False))
        completion = await self.llm.complete(prompt, task="optimize", payload=payload)
        return OptimizationAction.model_validate_json(completion.text)
