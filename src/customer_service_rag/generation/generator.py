from __future__ import annotations

import json
from importlib.resources import files

from jinja2 import Environment, StrictUndefined

from customer_service_rag.generation.citations import normalize_citations, validate_citations
from customer_service_rag.interfaces import LLMClient
from customer_service_rag.schemas import GeneratedResponse, Passage, QueryPlan


class ResponseGenerator:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def generate(self, query: QueryPlan, passages: list[Passage]) -> GeneratedResponse:
        payload = {
            "query": query.model_dump(mode="json"),
            "passages": [passage.model_dump(mode="json") for passage in passages],
        }
        aliases = {f"E{index}": passage.citation_id for index, passage in enumerate(passages, 1)}
        prompt_payload = {
            "question": query.current_message,
            "standalone_query": query.rewritten_query,
            "evidence": [{
                "id": f"E{index}", "title": passage.title, "text": passage.text,
            } for index, passage in enumerate(passages, 1)],
        }
        template = files("customer_service_rag").joinpath("prompts/generation.jinja2")
        prompt = Environment(undefined=StrictUndefined).from_string(
            template.read_text(encoding="utf-8")
        ).render(input_json=json.dumps(prompt_payload, ensure_ascii=False))
        completion = await self.llm.complete(prompt, task="generate", payload=payload)
        parsed = json.loads(completion.text)
        parsed["citations"] = normalize_citations(parsed.get("citations", []), aliases)
        response = GeneratedResponse.model_validate(parsed)
        validate_citations(response, passages)
        return response
