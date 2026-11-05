from __future__ import annotations

import json
from importlib.resources import files

from jinja2 import Environment, StrictUndefined

from customer_service_rag.interfaces import LLMClient
from customer_service_rag.schemas import (
    EvaluationFeedback,
    GeneratedResponse,
    Passage,
    QueryPlan,
)


class EvaluatorAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def evaluate(
        self, query: QueryPlan, passages: list[Passage], response: GeneratedResponse,
    ) -> EvaluationFeedback:
        payload = {
            "query": query.model_dump(mode="json"),
            "passages": [passage.model_dump(mode="json") for passage in passages],
            "response": response.model_dump(mode="json"),
        }
        template = files("customer_service_rag").joinpath("prompts/evaluation.jinja2")
        prompt = Environment(undefined=StrictUndefined).from_string(
            template.read_text(encoding="utf-8")
        ).render(input_json=json.dumps(payload, ensure_ascii=False))
        completion = await self.llm.complete(prompt, task="evaluate", payload=payload)
        return EvaluationFeedback.model_validate_json(completion.text)
