"""Part 2 原生 BGE/FAISS 连接最新 main；其他成员模块保持原样。"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from customer_service_rag.config import LLMConfig, PlanningConfig
from customer_service_rag.models.llm import build_llm
from customer_service_rag.part3 import EvidenceResponseGenerator
from customer_service_rag.planning.planner import QueryPlanner
from customer_service_rag.schemas import DialogueInput
from optimization_v2.adapters import LLMQueryRewriter, LLMRevisionGenerator, MainEvaluatorAdapter
from optimization_v2.budget import BudgetExceeded, BudgetLedger, BudgetedLLM
from optimization_v2.controller import OptimizationController
from optimization_v2.retrieval_adapter import Part2RetrievalAdapter
from optimization_v2.schemas import BudgetState, EvaluationRequest, Limits, OptimizationInput
from retrieval_v2.retrieval import HybridRetrievalPipeline, RetrievalConfig
from retrieval_v2.retrieval.corpus import load_passages_jsonl, passage_fingerprint


async def run_native(pipeline, dialogue, llm, *, corpus_version, top_k=3,
                     domain_filter=False, limits=None, maximum=512):
    """复用 main 的规划、生成、评价、优化；只更换检索资产与输出封装。"""
    limits = limits or Limits(escalation_enabled=False, max_total_tokens=30000)
    budget = BudgetLedger(limits, BudgetState())
    adapter = Part2RetrievalAdapter(pipeline, corpus_version=corpus_version)
    options = adapter.default_options()
    options = type(options).model_validate({
        **options.model_dump(), "final_top_k": top_k, "domain_filter": domain_filter,
    })
    bounded = BudgetedLLM(llm, budget, maximum)
    try:
        # 初次和优化阶段使用同一个 ledger，不重复计算或重置预算。
        plan = await budget.run(lambda: QueryPlanner(bounded, PlanningConfig(decompose=False)).plan(dialogue))
        snapshot = await budget.run(lambda: adapter.retrieve(plan, options, trace_id=dialogue.conversation_id))
        response = await budget.run(lambda: EvidenceResponseGenerator(bounded).generate(plan, snapshot.evidence))
        evaluator = MainEvaluatorAdapter(llm, max_output_tokens=maximum)
        feedback = await budget.run(lambda: evaluator.evaluate(EvaluationRequest(
            query=plan, evidence=snapshot.evidence, response=response,
        ), budget))
        controller = OptimizationController(
            retrieval=adapter, evaluator=evaluator,
            generator=LLMRevisionGenerator(llm, max_output_tokens=maximum),
            rewriter=LLMQueryRewriter(llm, max_output_tokens=maximum),
        )
        result = await controller.run(OptimizationInput(
            query=plan, response=response, evidence=snapshot.evidence, feedback=feedback,
            options=options, snapshot=snapshot, capabilities=adapter.capabilities,
            limits=limits, budget=BudgetState(used_tokens=budget.used_tokens,
                                            elapsed_seconds=budget.elapsed_seconds),
            trace_id=dialogue.conversation_id,
        ), budget=budget)
        output = result.model_dump(mode="json", exclude={"snapshot"})
        output.update({
            "initial_response": response.model_dump(mode="json"),
            "initial_evidence": [e.to_dict() for e in snapshot.evidence],
            "retrieval_rounds_total": 1 + result.retrieval_rounds,
            "evaluations_total": 1 + result.evaluations,
            "retrieval_backend": "bge_faiss_section", "corpus_version": corpus_version,
            "llm_backend": "mock" if llm.is_mock else "deepseek",
            "embedding_model": pipeline.config.embedding_model,
            "reranker_model": pipeline.config.reranker_model if options.rerank else None,
            "citation_scheme": "native_doc::section",
        })
        return output
    except Exception as error:
        return {
            "response": {"answer": "Unable to complete this request reliably.",
                         "citations": [], "status": "error"},
            "stop_reason": str(error) if isinstance(error, BudgetExceeded) else type(error).__name__,
            "used_tokens": budget.used_tokens, "elapsed_seconds": budget.elapsed_seconds,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["rrf", "full"], default="rrf")
    parser.add_argument("--llm", choices=["mock", "deepseek"], default="mock")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"))
    parser.add_argument("--top-k", type=int, choices=range(1, 21), default=3)
    parser.add_argument("--domain-filter", action="store_true")
    parser.add_argument("--history", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-retries", type=int, choices=range(3), default=2)
    args = parser.parse_args()
    if args.llm == "deepseek" and not os.environ.get("DEEPSEEK_API_KEY"):
        parser.error("Set DEEPSEEK_API_KEY before using --llm deepseek")
    config = RetrievalConfig.for_mode(args.mode, artifact_dir=args.artifact_dir,
                                      allow_reranker_fallback=False)
    pipeline = HybridRetrievalPipeline.from_artifacts(
        artifact_dir=args.artifact_dir, config=config, device=args.device,
    )
    llm = build_llm(LLMConfig(backend=args.llm,
        model="mock-extractive-v1" if args.llm == "mock" else args.model, max_new_tokens=512))
    dialogue = DialogueInput(
        conversation_id="native-main", current_message=args.query,
        dialogue_history=json.loads(args.history.read_text(encoding="utf-8")) if args.history else [],
    )
    output = asyncio.run(run_native(
        pipeline, dialogue, llm, top_k=args.top_k, domain_filter=args.domain_filter,
        corpus_version=passage_fingerprint(load_passages_jsonl(args.artifact_dir / "passages.jsonl")),
        limits=Limits(max_retries=args.max_retries, escalation_enabled=False,
                      max_total_tokens=30000, max_latency_seconds=180),
    ))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in (
        "response", "stop_reason", "retrieval_backend", "llm_backend",
        "retrieval_rounds_total", "evaluations_total", "used_tokens",
    ) if key in output}, ensure_ascii=False, indent=2))
    if output["response"]["status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
