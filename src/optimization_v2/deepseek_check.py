"""Explicit DeepSeek connectivity check using fictional evidence, never gold data."""

import argparse
import asyncio
import json
from pathlib import Path

from customer_service_rag.schemas import GeneratedResponse, QueryPlan
from retrieval_v2.retrieval.schemas import Evidence

from .budget import BudgetLedger
from .deepseek import DeepSeekLLM, create_deepseek_controller
from .schemas import BudgetState, Feedback, Limits, OptimizationInput, RewriteRequest


class UnusedRetrieval:
    async def retrieve(self, *args, **kwargs):
        raise RuntimeError("Connectivity check must only regenerate from supplied evidence")


async def run_check(client):
    controller = create_deepseek_controller(retrieval=UnusedRetrieval(), llm=client)
    limits = Limits(max_retries=1, max_latency_seconds=180, max_total_tokens=20000)
    budget = BudgetLedger(limits, BudgetState())
    query = QueryPlan(
        conversation_id="fictional-deepseek-check",
        current_message="For the fictional training permit, what do I bring and when must I apply?",
        dialogue_history=[], rewritten_query="fictional training permit requirements",
    )
    evidence = [Evidence(
        passage_id="fictional::permit-1", doc_id="fictional-permit", domain="unknown",
        title="Fictional training example only", section_title="Replacement",
        text="For this fictional training permit, bring identification and apply within 30 days.",
        span_ids=["1"], source_retrievers=[], matched_query=query.rewritten_query,
    )]
    response = GeneratedResponse(answer="Bring identification.", citations=[evidence[0].passage_id])
    feedback = Feedback(
        relevance=0.6, sufficiency=0.9, groundedness=0.9, dialogue_consistency=0.9,
        decision="retry", reason="The answer omits the deadline stated in the evidence.",
        missing_information="Include the 30-day deadline.",
        failure_types=["answer_defect"], evidence_sufficient=True,
    )
    rewrite_request = RewriteRequest(
        query=query, evidence=evidence, previous_response=response, feedback=feedback,
        options=OptimizationInput(
            query=query, evidence=evidence, response=response, feedback=feedback,
        ).options,
    )
    update = await budget.run(lambda: controller.rewriter.rewrite(rewrite_request, budget))
    query = QueryPlan.model_validate({**query.model_dump(), **update.model_dump()})
    result = await controller.run(OptimizationInput(
        query=query, evidence=evidence, response=response, feedback=feedback,
        limits=limits, budget=BudgetState(
            used_tokens=budget.used_tokens, elapsed_seconds=budget.elapsed_seconds,
        ),
    ), budget=budget)
    summary = {
        "is_mock": False, "model": client.model_name,
        "tasks": [{"task": c["task"], "status": c["status"]} for c in client.calls],
        "used_tokens": result.used_tokens, "status": result.response.status.value,
        "stop_reason": result.stop_reason,
        "rewritten_query": update.rewritten_query,
        "answer": result.response.answer, "citations": result.response.citations,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if result.response.status.value == "error":
        raise RuntimeError("DeepSeek optimizer check failed; see sanitized status above")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Send real DeepSeek requests")
    parser.add_argument("--config", type=Path, help="Reuse another existing local config")
    parser.add_argument("--model", help="Override the configured model")
    args = parser.parse_args()
    client = DeepSeekLLM(config_path=args.config, model=args.model)
    if not args.run:
        print(json.dumps({
            "status": "configured_not_sent", "model": client.model_name,
            "endpoint": ("https://api.deepseek.com/chat/completions"
                         if client._delegate is not None else client.transport.API_URL),
            "tasks": ["plan", "generate", "evaluate"],
        }, ensure_ascii=False, indent=2))
        return
    asyncio.run(run_check(client))


if __name__ == "__main__":
    main()
