"""Offline wiring demo for four branches. Not an answer-quality experiment."""

import asyncio
import json

from customer_service_rag.schemas import GeneratedResponse, QueryPlan
from retrieval_v2.retrieval.config import RetrievalConfig
from retrieval_v2.retrieval.pipeline import HybridRetrievalPipeline
from retrieval_v2.retrieval.schemas import Passage, SearchHit

from .controller import OptimizationController
from .retrieval_adapter import Part2RetrievalAdapter
from .schemas import Feedback, OptimizationInput, QueryUpdate


class DemoSearch:
    def __init__(self, name):
        self.name = name

    def search(self, query, *, domain=None, top_k=20):
        ids = ["a", "b", "c"] if self.name == "dense" else ["b", "a", "c"]
        return [SearchHit(
            Passage(
                passage_id=f"demo::{key}", doc_id="demo:dmv", domain="dmv",
                title="Fictional license policy", section_title=key, parent_titles=[],
                text=f"Fictional evidence {key}: bring identification within 30 days.",
                span_ids=[key],
            ), self.name, rank, 1 / rank, query,
        ) for rank, key in enumerate(ids[:top_k], 1)]


class DemoReranker:
    def rerank(self, main_query, sub_queries, candidates):
        for item in candidates:
            item.rerank_score = float(ord(item.passage.passage_id[-1]))
        return sorted(candidates, key=lambda item: -item.rerank_score)


class DemoGenerator:
    async def generate(self, request, budget):
        return GeneratedResponse(
            answer="Bring identification within 30 days. (Fictional demo policy.)",
            citations=[request.evidence[0].passage_id],
        )


class DemoRewriter:
    async def rewrite(self, request, budget):
        return QueryUpdate(rewritten_query="license replacement identification deadline")


class DemoEvaluator:
    async def evaluate(self, request, budget):
        return Feedback(
            relevance=0.9, sufficiency=0.9, groundedness=0.9,
            dialogue_consistency=0.9, decision="pass",
            reason="Scripted demo pass, not semantic quality validation",
        )


async def main():
    pipeline = HybridRetrievalPipeline(
        dense_retriever=DemoSearch("dense"), bm25_retriever=DemoSearch("bm25"),
        config=RetrievalConfig(dense_top_k=3, bm25_top_k=3, fused_top_k=3, final_top_k=2),
    )
    adapter = Part2RetrievalAdapter(
        pipeline, corpus_version="fictional-demo-v1",
        rerankers={"current_first_user_history": DemoReranker()},
    )
    controller = OptimizationController(
        retrieval=adapter, generator=DemoGenerator(), evaluator=DemoEvaluator(),
        rewriter=DemoRewriter(),
    )
    query = QueryPlan(
        conversation_id="demo", current_message="What do I bring and when?",
        rewritten_query="license replacement", domain="unknown",
    )
    options = adapter.default_options()
    initial = await adapter.retrieve(query, options)
    results = []
    for failure in ("answer_defect", "query_misunderstanding", "retrieval_balance", "evidence_ranking"):
        request = OptimizationInput(
            query=query, evidence=initial.evidence, snapshot=initial,
            response=GeneratedResponse(answer="Bring identification", citations=["demo::a"]),
            feedback=Feedback(
                relevance=0.9, sufficiency=0.9, groundedness=0.9, dialogue_consistency=0.9,
                decision="retry", reason="Demo missing deadline", failure_types=[failure],
                preferred_retriever="bm25", evidence_sufficient=True,
            ),
            options=options, capabilities=adapter.capabilities,
        )
        result = await controller.run(request)
        if result.stop_reason != "evaluation_passed":
            raise RuntimeError(result.model_dump_json())
        results.append({
            "failure_type": failure, "branch": result.actions[-1].branch,
            "status": result.response.status.value, "stop_reason": result.stop_reason,
            "new_retrieval_rounds": result.retrieval_rounds,
            "evaluations": result.evaluations,
            "answer": result.response.answer, "citations": result.response.citations,
        })
    print(json.dumps({"is_mock": True, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
