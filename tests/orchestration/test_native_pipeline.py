"""Native retrieval uses the same application runner and request budget."""

import json
import unittest

from customer_service_rag.config import AppConfig
from customer_service_rag.models.llm import MockLLM
from customer_service_rag.orchestration.orchestrator import RAGOrchestrator
from customer_service_rag.orchestration.retrieval import ProjectRetriever
from customer_service_rag.schemas import DialogueInput, LLMResult, TokenUsage
from customer_service_rag.retrieval import HybridRetrievalPipeline, Passage, RetrievalConfig
from customer_service_rag.retrieval.schemas import SearchHit


class Search:
    calls = 0

    def search(self, text, *, domain=None, top_k=10):
        self.calls += 1
        passage = Passage("policy:dmv::1", "policy:dmv", "dmv", "License", "Replace",
                          [], "Replace a lost license with proof of identification.", ["1"])
        return [SearchHit(passage, "bm25", 1, 1.0, text)]


class RetryLLM(MockLLM):
    def __init__(self):
        super().__init__()
        self.evaluations = 0

    async def complete(self, prompt, *, task, payload):
        if task != "evaluate":
            return await super().complete(prompt, task=task, payload=payload)
        self.evaluations += 1
        retry = self.evaluations == 1
        return LLMResult(text=json.dumps({
            "relevance": .9, "sufficiency": .4 if retry else .9,
            "groundedness": .9, "dialogue_consistency": .9,
            "decision": "retry" if retry else "pass", "reason": "fixed test",
            "failure_types": ["answer_defect"] if retry else [],
            "evidence_sufficient": True,
        }), model="test", is_mock=True,
            usage=TokenUsage(input_tokens=2, output_tokens=2, source="counted"))


class NativePipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.search = Search()
        self.pipeline = HybridRetrievalPipeline(
            dense_retriever=None, bm25_retriever=self.search,
            config=RetrievalConfig.for_mode("bm25"),
        )
        self.dialogue = DialogueInput(conversation_id="t", current_message="Replace my license")
        self.config = AppConfig(
            retrieval={"backend": "native", "branches": ["bm25"], "final_top_k": 1},
            quality={"enabled": True}, optimization={"enabled": True},
            budget={"max_total_tokens": 30000, "max_latency_seconds": 180},
        )

    def runner(self, llm, config=None):
        config = config or self.config
        retriever = ProjectRetriever.from_pipeline(config, self.pipeline, corpus_version="test")
        return RAGOrchestrator(config, retriever=retriever, llm_factory=lambda: llm)

    async def test_native_ids_survive_generation_and_evaluation(self):
        llm = MockLLM()
        result = await self.runner(llm).run(self.dialogue)
        self.assertEqual(result.stop_reason, "evaluation_passed")
        self.assertEqual(result.response.citations, ["policy:dmv::1"])
        self.assertEqual(result.integration["used_tokens"], sum(u.total_tokens for u in llm.usage))
        self.assertEqual(self.search.calls, 1)

    async def test_main_controller_regenerates_without_recall(self):
        llm = RetryLLM()
        result = await self.runner(llm).run(self.dialogue)
        self.assertEqual(result.stop_reason, "evaluation_passed")
        self.assertEqual(result.actions[0].action, "regenerate")
        self.assertEqual(result.iteration, 2)
        self.assertEqual(result.retrieval_round, 1)
        self.assertEqual(self.search.calls, 1)

    async def test_initial_budget_is_not_reset(self):
        config = self.config.model_copy(update={
            "budget": self.config.budget.model_copy(update={"max_total_tokens": 1}),
        })
        result = await self.runner(MockLLM(), config).run(self.dialogue)
        self.assertEqual(result.stop_reason, "token_budget")
        self.assertEqual(self.search.calls, 0)


if __name__ == "__main__":
    unittest.main()
