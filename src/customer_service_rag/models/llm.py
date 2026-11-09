"""Mock structured completion; no remote calls and no LLM quality claims."""

from __future__ import annotations

import json
import math
from typing import Any, Callable
from pathlib import Path

from customer_service_rag.config import LLMConfig
from customer_service_rag.models.mock_logic import evaluate_payload
from customer_service_rag.schemas import LLMResult, TokenUsage


class MockLLM:
    model_name = "mock-extractive-v1"
    is_mock = True

    def __init__(self, usage_guard: Callable[[TokenUsage], None] | None = None) -> None:
        self.usage: list[TokenUsage] = []
        self.usage_guard = usage_guard
        self.calls: list[dict] = []

    async def complete(self, prompt: str, *, task: str, payload: dict[str, Any]) -> LLMResult:
        if task == "plan":
            history = payload.get("dialogue_history", [])
            text = payload["current_message"]
            if payload.get("rewrite") and history:
                context = " ".join(item["text"] for item in history)
                text = f"Conversation context: {context}\nCurrent question: {text}"
            value = {
                "rewritten_query": text,
                "intent": "unknown",
                "domain": "unknown",
                "sub_queries": [],
            }
        elif task == "generate":
            passages = payload["passages"]
            if not passages:
                value = {
                    "answer": "The available evidence is insufficient to answer this question.",
                    "citations": [],
                    "status": "insufficient_evidence",
                }
            else:
                passage = passages[0]
                value = {
                    "answer": passage["text"],
                    "citations": [f'{passage["doc_id"]}:{passage["passage_id"]}'],
                    "status": "answered",
                }
        elif task == "evaluate":
            value = evaluate_payload(payload)
        else:
            raise NotImplementedError(f"No mock completion implemented for task {task!r}")
        text = json.dumps(value, ensure_ascii=False)
        usage = TokenUsage(
            input_tokens=max(1, math.ceil(len(prompt) / 4)),
            output_tokens=max(1, math.ceil(len(text) / 4)),
            source="estimated",
        )
        if self.usage_guard is not None:
            self.usage_guard(usage)
        self.usage.append(usage)
        self.calls.append({
            "task": task, "model": self.model_name, "is_mock": True,
            "usage": usage.model_dump(mode="json"), "raw_text": text,
        })
        return LLMResult(text=text, model=self.model_name, is_mock=True, usage=usage)


def build_llm(
    config: LLMConfig, usage_guard: Callable[[TokenUsage], None] | None = None,
    root: Path | None = None,
):
    if config.backend == "transformers":
        from customer_service_rag.models.local_llm import LocalLLM
        return LocalLLM(config, root or Path.cwd(), usage_guard=usage_guard)
    if config.backend == "deepseek":
        from customer_service_rag.models.deepseek_llm import DeepSeekLLM
        return DeepSeekLLM(config, usage_guard=usage_guard)
    return MockLLM(usage_guard=usage_guard)
