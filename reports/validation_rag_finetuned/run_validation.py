"""Official validation: original dialogue -> fine-tuned Dense -> DeepSeek.

This experiment has no planner, domain prediction, subqueries, BM25 or reranker.
The retrieval query is main's deterministic finetuned_query representation of
the visible dialogue, matching the model's training input format.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAIN = ROOT / "main"
PREVIOUS = ROOT / "validation_b_finetuned"
sys.path[:0] = [str(PREVIOUS / "vendor"), str(MAIN / "src"), str(PREVIOUS)]

from customer_service_rag.config import load_config
from customer_service_rag.data.io import load_examples
from customer_service_rag.evaluation.response_metrics import response_scores
from customer_service_rag.evaluation.retrieval_metrics import evidence_scores
from customer_service_rag.schemas import QueryPlan, ResponseStatus
from retrieval.corpus import load_passages_jsonl, passage_fingerprint
from retrieval.native_main import finetuned_query
from retrieval.pipeline import HybridRetrievalPipeline
from retrieval.profiles import profile_config
from retrieval.schemas import RetrievalRequest
from run_validation import append_jsonl, file_sha256, generate, hit_record

EXAMPLES = PREVIOUS / "data/multidoc2dial/validation.jsonl"
DOCUMENTS = ROOT / "multidoc2dial/data/multidoc2dial/multidoc2dial_doc.json"
INDEX = PREVIOUS / "artifacts/indexes/struct/finetuned"
OLD_EVALUATION = PREVIOUS / "results/validation_20261109/evaluation.jsonl"


def read_jsonl(path: Path):
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def plan_without_rewriting(example) -> dict:
    query_text = finetuned_query(example.dialogue)
    query = QueryPlan.model_validate({
        **example.dialogue.model_dump(mode="json"),
        "rewritten_query": query_text,
        "domain": "unknown", "sub_queries": [], "intent": "unknown",
    })
    return {
        "query_plan": query.model_dump(mode="json"),
        "query_source": "main retrieval.native_main.finetuned_query",
        "rewrite_model_used": False, "predicted_domains": [],
        "retrieval_domains": [], "routing_status": "disabled",
        "usage": [], "model_calls": [], "requests": [],
    }


def retrieve_global(pipeline, planning: dict, top_k: int) -> dict:
    query = QueryPlan.model_validate(planning["query_plan"])
    started = perf_counter()
    hits = pipeline.dense_retriever.search(query.rewritten_query, domain=None,
                                           top_k=pipeline.config.dense_top_k)
    candidates = pipeline.dense_candidates([hits])[:pipeline.config.fused_top_k]
    # Call the public main pipeline for the evidence that enters generation.
    evidence = pipeline.retrieve(RetrievalRequest(
        query=query.rewritten_query, sub_queries=[], domain=None, top_k=top_k,
        trace_id=query.conversation_id,
    ))
    if [item.passage_id for item in evidence] != [item.passage.passage_id for item in candidates[:top_k]]:
        raise AssertionError("Public fine-tuned retrieval disagrees with saved Dense ranking")
    elapsed = perf_counter() - started
    return {
        "query_used": query.rewritten_query,
        "domain_filter": None, "sub_queries_used": [],
        "options": {"dense_top_k": pipeline.config.dense_top_k,
                    "candidate_top_k": pipeline.config.fused_top_k,
                    "final_top_k": top_k, "use_bm25": False,
                    "use_sub_queries": False, "use_reranker": False},
        "rankings": [[hit_record(hit) for hit in hits]],
        "candidates": [
            {"passage_id": item.passage.passage_id, "doc_id": item.passage.doc_id,
             "domain": item.passage.domain, "dense_score": item.rerank_score}
            for item in candidates
        ],
        "evidence": [item.to_dict() for item in evidence],
        "elapsed_seconds": elapsed,
    }


def compare_runs(output: Path, current: dict) -> dict:
    old = {row["sample_id"]: row for row in read_jsonl(OLD_EVALUATION)}
    new = {row["sample_id"]: row for row in read_jsonl(output / "evaluation.jsonl")}
    if set(old) != set(new):
        return {"complete_pairing": False, "old_count": len(old), "new_count": len(new)}
    result = {"complete_pairing": True, "sample_count": len(new), "metrics": {}}
    for name, get in (
        ("response_f1", lambda row: row["response"]["f1"]),
        ("evidence_recall_at_10", lambda row: row["retrieval"]["@10"]["recall"]),
    ):
        deltas = [get(new[sid]) - get(old[sid]) for sid in new]
        result["metrics"][name] = {
            "old_mean": sum(get(row) for row in old.values()) / len(old),
            "new_mean": sum(get(row) for row in new.values()) / len(new),
            "mean_delta": sum(deltas) / len(deltas),
            "new_better": sum(value > 1e-12 for value in deltas),
            "equal": sum(abs(value) <= 1e-12 for value in deltas),
            "new_worse": sum(value < -1e-12 for value in deltas),
        }
    (output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def evaluate_streaming(cases_path: Path, examples: dict, passages, output: Path) -> dict:
    by_span = defaultdict(list)
    for passage in passages:
        for span_id in passage.span_ids:
            by_span[(passage.doc_id, str(span_id))].append(passage.passage_id)
    totals = defaultdict(float)
    counts = Counter()
    status_counts = Counter()
    scored_count = 0
    sample_count = 0

    def add(name, value):
        if value is not None:
            totals[name] += value
            counts[name] += 1

    with (output / "evaluation.jsonl").open("w", encoding="utf-8") as stream:
        for case in read_jsonl(cases_path):
            example = examples[case["sample_id"]]
            groups = []
            for reference in example.grounding:
                for span_id in reference.span_ids:
                    group = by_span.get((reference.original_doc_id, str(span_id)), [])
                    if group and group not in groups:
                        groups.append(group)
            ranking = [item["passage_id"] for item in case["retrieval"]["candidates"]]
            answer = case["generation"]["answer"]
            citations = set(answer["citations"])
            evidence_ids = {item["passage_id"] for item in case["retrieval"]["evidence"]}
            gold_ids = {item for group in groups for item in group}
            item = {
                "sample_id": case["sample_id"], "origin_split": example.origin_split,
                "gold_domain": example.domain, "reference_answers": example.reference_answers,
                "gold_grounding": [g.model_dump(mode="json") for g in example.grounding],
                "gold_struct_groups": groups,
                "response": response_scores(answer["answer"], example.reference_answers,
                                            answer["status"] == ResponseStatus.ANSWERED.value),
                "retrieval": {f"@{k}": evidence_scores(ranking, groups, k) for k in (1, 5, 10)},
                "citation_evidence_validity": len(citations & evidence_ids) / len(citations) if citations else None,
                "citation_gold_precision": len(citations & gold_ids) / len(citations) if citations else None,
            }
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            sample_count += 1
            scored_count += bool(groups)
            status_counts[answer["status"]] += 1
            add("response_f1", item["response"]["f1"])
            add("response_em", item["response"]["em"])
            for k in (1, 5, 10):
                for metric in ("recall", "hit", "complete_coverage"):
                    add(f"{k}_{metric}", item["retrieval"][f"@{k}"][metric])
            add("citation_evidence_validity", item["citation_evidence_validity"])
            add("citation_gold_precision", item["citation_gold_precision"])

    def avg(name):
        return totals[name] / counts[name] if counts[name] else None

    return {
        "sample_count": sample_count,
        "status_counts": dict(status_counts),
        "response_f1": avg("response_f1"), "response_em": avg("response_em"),
        "retrieval": {f"@{k}": {metric: avg(f"{k}_{metric}")
                                  for metric in ("recall", "hit", "complete_coverage")}
                      for k in (1, 5, 10)},
        "retrieval_scored_count": scored_count,
        "citation_evidence_validity": avg("citation_evidence_validity"),
        "citation_gold_precision": avg("citation_gold_precision"),
    }


async def run(args) -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("Set DEEPSEEK_API_KEY in this process before running")
    import faiss
    import torch
    faiss.omp_set_num_threads(4)
    torch.set_num_threads(4)
    config = load_config(MAIN / "configs/integrated.yaml")
    examples_list = load_examples(EXAMPLES)
    examples = {item.sample_id: item for item in examples_list}
    selected = examples_list[:args.limit] if args.limit else examples_list
    checkpoint = MAIN / "models/bge-base-en-v1.5-struct-ft/checkpoint-2000"
    profile = profile_config(segmentation="struct", chain="finetuned",
                             artifact_dir=INDEX, embedding_model=checkpoint.as_posix())
    pipeline = HybridRetrievalPipeline.from_artifacts(
        artifact_dir=INDEX, config=profile, device=args.device)
    if profile.use_sub_queries or profile.use_bm25 or profile.use_reranker:
        raise AssertionError("Expected the fine-tuned Dense-only profile")
    passages = load_passages_jsonl(INDEX / "passages.jsonl")
    output = args.run_dir or HERE / "results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "MultiDoc2Dial official validation", "sample_count": len(examples_list),
        "examples_sha256": file_sha256(EXAMPLES),
        "documents_sha256": file_sha256(DOCUMENTS),
        "index_manifest_sha256": file_sha256(INDEX / "service_manifest.json"),
        "checkpoint_manifest_sha256": file_sha256(MAIN / "models/bge-base-en-v1.5-struct-ft/checkpoint_manifest.json"),
        "corpus_version": passage_fingerprint(passages),
        "query_source": "main retrieval.native_main.finetuned_query, no LLM rewrite",
        "domain_filter": None, "sub_queries": [], "top_k": args.top_k,
        "retrieval_chain": "struct fine-tuned Dense, no BM25, no reranker",
        "generation": "main EvidenceResponseGenerator, DeepSeek Flash",
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Run manifest differs; choose another output directory")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    cases_path, events_path = output / "cases.jsonl", output / "events.jsonl"
    completed = {row["sample_id"] for row in read_jsonl(cases_path)}
    phases = {}
    for event in read_jsonl(events_path):
        if event["sample_id"] not in completed:
            phases.setdefault(event["sample_id"], {})[event["phase"]] = event["data"]
    semaphore = asyncio.Semaphore(args.concurrency)

    async def process_one(example):
        async with semaphore:
            if example.sample_id in completed:
                return
            started = perf_counter()
            saved = phases.get(example.sample_id, {})
            planning = saved.get("planned")
            if planning is None:
                planning = plan_without_rewriting(example)
                append_jsonl(events_path, {"sample_id": example.sample_id, "phase": "planned", "data": planning})
            retrieval = saved.get("retrieved")
            if retrieval is None:
                retrieval = retrieve_global(pipeline, planning, args.top_k)
                append_jsonl(events_path, {"sample_id": example.sample_id, "phase": "retrieved", "data": retrieval})
            generation = saved.get("generated")
            if generation is None:
                generation = await generate(config, planning, retrieval)
                append_jsonl(events_path, {"sample_id": example.sample_id, "phase": "generated", "data": generation})
            row = {"sample_id": example.sample_id,
                   "question": example.dialogue.current_message,
                   "planning": planning, "retrieval": retrieval, "generation": generation,
                   "elapsed_seconds": perf_counter() - started}
            append_jsonl(cases_path, row)
            completed.add(example.sample_id)
            phases.pop(example.sample_id, None)
            if len(completed) % 25 == 0 or len(completed) == len(selected):
                print(f"{len(completed)}/{len(selected)} completed", flush=True)

    pending = [example for example in selected if example.sample_id not in completed]
    for start in range(0, len(pending), args.concurrency * 5):
        await asyncio.gather(*(process_one(example) for example in pending[start:start + args.concurrency * 5]))
    del pipeline
    gc.collect()
    report = evaluate_streaming(cases_path, examples, passages, output)
    report["query_source"] = manifest["query_source"]
    report["domain_filter"] = None
    report["note"] = "Official validation is a development split. No planner or domain filter; factual support needs separate labels."
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison = compare_runs(output, report) if len(completed) == len(examples_list) else None
    print(json.dumps({"run_dir": str(output), "metrics": report,
                      "comparison": comparison}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.top_k <= 20 or not 1 <= args.concurrency <= 16:
        parser.error("top-k must be 1..20 and concurrency must be 1..16")
    asyncio.run(run(args))
