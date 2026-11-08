"""Continue the saved A/B/C answers through main's evaluator and optimizer."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "vendor"))
sys.path.insert(0, str(ROOT / "main/src"))

from customer_service_rag.models.llm import build_llm
from customer_service_rag.orchestration.integrated_retrieval import IntegratedRetriever
from customer_service_rag.schemas import GeneratedResponse, QueryPlan
from optimization_v2.adapters import LLMQueryRewriter, LLMRevisionGenerator, MainEvaluatorAdapter
from optimization_v2.budget import BudgetLedger
from optimization_v2.controller import OptimizationController
from optimization_v2.schemas import (
    BudgetState, EvaluationRequest, Limits, OptimizationInput, RetrievalSnapshot,
)
from retrieval.schemas import Evidence

from prepare_main import model_config
from run_main_experiment import evidence_record, score_retrieval, summarize_main, usage


SOURCE = HERE / "results/20261107T194151Z"
ANNOTATIONS = HERE / "data/cross_domain_20_annotations.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_evidence(items, query):
    return [Evidence(
        passage_id=e["passage_id"], doc_id=e["doc_id"], domain=e["domain"],
        title=e["title"], section_title="", text=e["text"], span_ids=e["span_ids"],
        source_retrievers=e["source_retrievers"], matched_query=query.rewritten_query,
        rerank_score=e["rerank_score"],
    ) for e in items]


def initial_tokens(call):
    return sum(u["total_tokens"] or 0 for u in call["usage"])


def model_limits(config):
    return Limits(
        max_retries=config.optimization.max_retries if config.optimization.enabled else 0,
        max_total_tokens=config.budget.max_total_tokens,
        max_latency_seconds=config.budget.max_latency_seconds,
        escalation_enabled=config.escalation.enabled,
        **{name: getattr(config.quality, name) for name in (
            "relevance_threshold", "sufficiency_threshold", "groundedness_threshold",
            "dialogue_consistency_threshold",
        )},
    )


class ScopedRetrieval:
    """Keep a branch in its assigned corpus slice on every optimization retry."""

    def __init__(self, adapter, domain, cap):
        self.adapter = adapter
        self.domain = domain
        self.cap = cap
        self.capabilities = adapter.capabilities

    async def retrieve(self, query, options, snapshot=None, trace_id=None):
        scoped = query.model_copy(update={"domain": self.domain or "unknown"})
        bounded = options.model_copy(update={
            "domain_filter": self.domain is not None,
            "final_top_k": min(options.final_top_k, self.cap),
        })
        result = await self.adapter.retrieve(scoped, bounded, snapshot, trace_id)
        if self.domain is not None and any(e.domain != self.domain for e in result.evidence):
            raise AssertionError("Optimizer crossed a domain boundary")
        return result


class MultiDomainRetrieval:
    """Re-run filtered retrieval per predicted domain, then merge capped evidence."""

    def __init__(self, adapter, domains, cap=6):
        self.adapter = adapter
        self.domains = domains
        self.cap = cap
        self.capabilities = adapter.capabilities

    async def retrieve(self, query, options, snapshot=None, trace_id=None):
        total = min(options.final_top_k, self.cap)
        quotas = {d: total // len(self.domains) + (i < total % len(self.domains))
                  for i, d in enumerate(self.domains)}
        branches = {}
        for domain in self.domains:
            if not quotas[domain]:
                continue
            n = len(self.domains)
            scoped = query.model_copy(update={"domain": domain})
            per_domain = options.model_copy(update={
                "dense_top_k": (max(1, options.dense_top_k // n)
                                if options.dense_top_k else 0),
                "bm25_top_k": (max(1, options.bm25_top_k // n)
                               if options.bm25_top_k else 0),
                "candidate_top_k": max(quotas[domain], options.candidate_top_k // n),
                "final_top_k": quotas[domain], "domain_filter": True,
            })
            branches[domain] = await self.adapter.retrieve(scoped, per_domain, None, trace_id)
            if any(e.domain != domain for e in branches[domain].evidence):
                raise AssertionError("B optimizer crossed a domain boundary")
        longest = max((len(s.evidence) for s in branches.values()), default=0)
        evidence = [branches[d].evidence[i] for i in range(longest)
                    for d in self.domains if d in branches and i < len(branches[d].evidence)]
        return RetrievalSnapshot(
            recall_key=hashlib.sha256(json.dumps({
                "query": query.rewritten_query, "domains": self.domains,
                "options": options.model_dump(mode="json"),
            }, sort_keys=True).encode()).hexdigest(),
            corpus_version=self.adapter.corpus_version,
            evidence=evidence, reused_recall=False,
            elapsed_ms=sum(s.elapsed_ms for s in branches.values()),
            notes=["Filtered retrieval executed independently for each predicted domain"],
        )


class FixedEvidenceRetrieval:
    """Final C revision can use merged domain evidence but cannot fetch new data."""

    def __init__(self, adapter, evidence):
        self.capabilities = adapter.capabilities
        self.corpus_version = adapter.corpus_version
        self.evidence = evidence

    async def retrieve(self, query, options, snapshot=None, trace_id=None):
        return RetrievalSnapshot(
            recall_key="C-final-fixed-evidence", corpus_version=self.corpus_version,
            evidence=self.evidence, reused_recall=True,
            notes=["Final synthesis reuses completed domain evidence"],
        )


async def continue_one(config, query, initial, evidence, options, retrieval, label):
    llm = build_llm(config.models.llm, root=config.project_root)
    maximum = config.models.llm.max_new_tokens
    limits = model_limits(config)
    budget = BudgetLedger(limits, BudgetState(
        used_tokens=initial_tokens(initial), elapsed_seconds=initial["elapsed_seconds"],
    ))
    evaluator = MainEvaluatorAdapter(llm, max_output_tokens=maximum)
    response = GeneratedResponse.model_validate(initial["answer"])
    response.retrieval_round = 1
    started = perf_counter()
    try:
        feedback = await budget.run(lambda: evaluator.evaluate(EvaluationRequest(
            query=query, evidence=evidence, response=response,
        ), budget))
        controller = OptimizationController(
            retrieval=retrieval, evaluator=evaluator,
            generator=LLMRevisionGenerator(llm, max_output_tokens=maximum),
            rewriter=LLMQueryRewriter(llm, max_output_tokens=maximum),
        )
        result = await controller.run(OptimizationInput(
            query=query, response=response, evidence=evidence, feedback=feedback,
            options=options, capabilities=retrieval.capabilities, limits=limits,
            budget=BudgetState(used_tokens=budget.used_tokens,
                               elapsed_seconds=budget.elapsed_seconds),
            trace_id=label,
        ), budget=budget)
        final_response = result.response
        final_evidence = result.evidence
        workflow = {
            "initial_feedback": feedback.model_dump(mode="json"),
            "feedback_history": [x.model_dump(mode="json") for x in result.feedback_history],
            "actions": [x.model_dump(mode="json") for x in result.actions],
            "trace": [x.model_dump(mode="json") for x in result.trace],
            "retry_count": result.retry_count, "evaluations": 1 + result.evaluations,
            "retrieval_rounds_added": result.retrieval_rounds,
            "stop_reason": result.stop_reason,
        }
    except Exception as exc:
        final_response, final_evidence = response, evidence
        workflow = {"stop_reason": f"initial_evaluation_error:{type(exc).__name__}",
                    "error": str(exc), "evaluations": 0, "retry_count": 0,
                    "feedback_history": [], "actions": [], "trace": []}
    new_usage = usage(llm)
    generated = [u for u, call in zip(new_usage, llm.calls) if call["task"] == "generate"]
    return {
        "answer": final_response.model_dump(mode="json"),
        "evidence": [evidence_record(e) for e in final_evidence],
        "generation_usage": initial["usage"] + generated,
        "pipeline_usage": initial["usage"] + new_usage,
        "new_calls": llm.calls,
        "workflow": workflow,
        "added_elapsed_seconds": perf_counter() - started,
    }


def packed_generation(continued, initial):
    return {"answer": continued["answer"], "error": None,
            "usage": continued["generation_usage"],
            "elapsed_seconds": initial["elapsed_seconds"] + sum(
                call.get("elapsed_seconds", 0) for call in continued["new_calls"]
                if call["task"] == "generate")}


async def run(args):
    config = model_config()
    config = config.model_copy(update={"retrieval": config.retrieval.model_copy(
        update={"final_top_k": 6, "candidate_top_k": 30})})
    if not config.quality.enabled or not config.optimization.enabled:
        raise ValueError("main quality and optimization must both be enabled")
    index = config.resolve_path(config.paths.index_dir) / "manifest.json"
    if not index.is_file():
        raise FileNotFoundError("Run prepare_main.py before the full pipeline")
    retriever = IntegratedRetriever(config)
    source = args.source_dir
    rows = [json.loads(line) for line in (source / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    samples = {s["sample_id"]: s for s in json.loads(ANNOTATIONS.read_text(encoding="utf-8"))}
    output = args.run_dir or HERE / "results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_full")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest["corpus_sha256"] != json.loads(index.read_text(encoding="utf-8"))["corpus_sha256"]:
        raise ValueError("Source answers and current index use different corpora")
    expected = {"source_batch": str(source.relative_to(HERE)),
                "source_cases_sha256": sha(source / "cases.jsonl"),
                "annotation_sha256": sha(ANNOTATIONS),
                "config": "main/configs/integrated.yaml", "max_retries": config.optimization.max_retries,
                "max_total_tokens": config.budget.max_total_tokens,
                "max_latency_seconds": config.budget.max_latency_seconds,
                "evidence_cap": 6, "reuse_saved_initial_answers": True}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != expected:
            raise ValueError("Cannot resume a batch with different inputs or settings")
    else:
        manifest_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    result_path = output / "cases.jsonl"
    completed = ({json.loads(line)["sample_id"] for line in result_path.read_text(encoding="utf-8").splitlines()}
                 if result_path.exists() else set())
    with result_path.open("a", encoding="utf-8") as stream:
        for old in rows[:args.limit or None]:
            sid = old["sample_id"]
            if sid in completed:
                continue
            query = QueryPlan.model_validate(old["query_plan"])
            domains = old["predicted_domains"]
            sample = samples[sid]
            result_row = {k: old[k] for k in (
                "sample_id", "question", "predicted_domains", "gold_domains",
                "query_plan", "planner_usage",
            )}
            a_options = retriever.options(config.retrieval)
            a = await continue_one(config, query, old["A"]["generation"],
                as_evidence(old["A"]["evidence"], query), a_options,
                ScopedRetrieval(retriever.adapter, None, 6), sid + ":A")
            result_row["A"] = {"evidence": a["evidence"],
                "generation": packed_generation(a, old["A"]["generation"]),
                "pipeline_usage": a["pipeline_usage"], "workflow": a["workflow"],
                "added_elapsed_seconds": a["added_elapsed_seconds"],
                "retrieval_metrics": score_retrieval(a["evidence"], sample)}
            b = await continue_one(config, query, old["B"]["generation"],
                as_evidence(old["B"]["evidence"], query), a_options,
                MultiDomainRetrieval(retriever.adapter, domains), sid + ":B")
            result_row["B"] = {"evidence": b["evidence"],
                "generation": packed_generation(b, old["B"]["generation"]),
                "pipeline_usage": b["pipeline_usage"], "workflow": b["workflow"],
                "added_elapsed_seconds": b["added_elapsed_seconds"],
                "retrieval_metrics": score_retrieval(b["evidence"], sample)}
            quotas = {d: 6 // len(domains) + (i < 6 % len(domains))
                      for i, d in enumerate(domains)}
            c_subanswers, c_evidence, c_pipeline_usage = {}, [], []
            for domain in domains:
                scoped = query.model_copy(update={"domain": domain,
                    "current_message": f"Answer only the {domain} part of this request: {old['question']}"})
                old_call = old["C"]["subanswers"][domain]
                old_evidence = old["C"]["branches"][domain]["evidence"]
                options = a_options.model_copy(update={
                    "dense_top_k": max(1, 50 // len(domains)),
                    "bm25_top_k": max(1, 50 // len(domains)),
                    "candidate_top_k": max(quotas[domain], 30 // len(domains)),
                    "final_top_k": quotas[domain], "domain_filter": True,
                })
                sub = await continue_one(config, scoped, old_call,
                    as_evidence(old_evidence, scoped), options,
                    ScopedRetrieval(retriever.adapter, domain, quotas[domain]),
                    sid + ":C:" + domain)
                if any(e["domain"] != domain for e in sub["evidence"]):
                    raise AssertionError("C subanswer has cross-domain evidence")
                c_subanswers[domain] = {"answer": sub["answer"],
                    "usage": sub["generation_usage"],
                    "pipeline_usage": sub["pipeline_usage"],
                    "evidence": sub["evidence"], "workflow": sub["workflow"],
                    "added_elapsed_seconds": sub["added_elapsed_seconds"]}
                c_pipeline_usage.extend(sub["pipeline_usage"])
            longest = max((len(s["evidence"]) for s in c_subanswers.values()), default=0)
            c_evidence = [c_subanswers[d]["evidence"][i] for i in range(longest)
                          for d in domains if i < len(c_subanswers[d]["evidence"])]
            summary = await summarize_main(config, old["question"],
                {d: c_subanswers[d]["answer"] for d in domains},
                as_evidence(c_evidence, query))
            final = await continue_one(config, query, summary,
                as_evidence(c_evidence, query), a_options,
                FixedEvidenceRetrieval(retriever.adapter, as_evidence(c_evidence, query)),
                sid + ":C:final")
            c_pipeline_usage.extend(final["pipeline_usage"])
            result_row["C"] = {"evidence": final["evidence"],
                "subanswers": c_subanswers,
                "generation": packed_generation(final, summary),
                "pipeline_usage": c_pipeline_usage, "workflow": final["workflow"],
                "added_elapsed_seconds": final["added_elapsed_seconds"]
                    + summary["elapsed_seconds"]
                    + sum(s["added_elapsed_seconds"] for s in c_subanswers.values()),
                "retrieval_metrics": score_retrieval(final["evidence"], sample)}
            stream.write(json.dumps(result_row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{len(completed)+1:02d}/{len(rows)} {sid} " + " ".join(
                f"{arm}={result_row[arm]['workflow']['stop_reason']}"
                for arm in "ABC"), flush=True)
            completed.add(sid)
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--limit", type=int)
    asyncio.run(run(parser.parse_args()))
