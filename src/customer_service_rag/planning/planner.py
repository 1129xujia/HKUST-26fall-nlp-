from __future__ import annotations

import json
from importlib.resources import files

from jinja2 import Environment, StrictUndefined

from customer_service_rag.config import PlanningConfig
from customer_service_rag.interfaces import LLMClient
from customer_service_rag.planning.decomposition import normalize_sub_queries
from customer_service_rag.planning.rewriting import normalize_query
from customer_service_rag.planning.routing import normalize_domain, route_visible_dialogue
from customer_service_rag.schemas import DialogueInput, QueryPlan


class QueryPlanner:
    def __init__(self, llm: LLMClient, config: PlanningConfig) -> None:
        self.llm = llm
        self.config = config

    async def plan(self, dialogue: DialogueInput) -> QueryPlan:
        payload = dialogue.model_dump(mode="json")
        if not self.config.use_history:
            payload["dialogue_history"] = []
        payload["rewrite"] = self.config.rewrite
        payload["decompose"] = self.config.decompose
        payload["max_sub_queries"] = self.config.max_sub_queries
        if not self.config.rewrite:
            query = dialogue.current_message
            if self.config.use_history and payload["dialogue_history"]:
                query = " ".join(item["text"] for item in payload["dialogue_history"]) + " " + query
            return QueryPlan.model_validate({
                **dialogue.model_dump(mode="json"),
                "dialogue_history": payload["dialogue_history"], "rewritten_query": query,
            })
        template = files("customer_service_rag").joinpath("prompts/planning.jinja2")
        prompt = Environment(undefined=StrictUndefined).from_string(
            template.read_text(encoding="utf-8")
        ).render(input_json=json.dumps(payload, ensure_ascii=False))
        completion = await self.llm.complete(prompt, task="plan", payload=payload)
        result = json.loads(completion.text)
        result["rewritten_query"] = normalize_query(result["rewritten_query"])
        result["domain"] = normalize_domain(result.get("domain", "unknown"))
        if result["domain"] == "unknown":
            result["domain"] = route_visible_dialogue(
                dialogue.current_message, payload["dialogue_history"],
            )
        result["sub_queries"] = (
            normalize_sub_queries(
                result.get("sub_queries", []), result["rewritten_query"], self.config.max_sub_queries,
            ) if self.config.decompose else []
        )
        return QueryPlan.model_validate({
            **dialogue.model_dump(mode="json"),
            "dialogue_history": payload["dialogue_history"],
            **result,
        })
