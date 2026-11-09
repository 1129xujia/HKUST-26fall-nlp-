"""Evaluate a service profile against frozen query/gold JSON samples."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from experiments import ROOT
from customer_service_rag.retrieval.profiles import profile_config
from customer_service_rag.retrieval.pipeline import HybridRetrievalPipeline
from customer_service_rag.retrieval.schemas import RetrievalRequest


def ranking_metrics(ranking, gold, cutoffs=(5, 10)):
    gold = set(gold)
    if not gold or len(ranking) != len(set(ranking)):
        raise ValueError("Gold must be nonempty and ranking IDs must be unique")
    values = {}
    for k in cutoffs:
        relevance = [int(pid in gold) for pid in ranking[:k]]
        first = next((i for i, relevant in enumerate(relevance, 1) if relevant), None)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
        values.update({f"hit@{k}": float(any(relevance)),
                       f"recall@{k}": sum(relevance) / len(gold),
                       f"mrr@{k}": 1 / first if first else 0.,
                       f"ndcg@{k}": sum(r / math.log2(i + 2) for i, r in enumerate(relevance)) / ideal})
    return values


def summarize(rows):
    return {"n": len(rows),
            **{key: float(np.mean([r["metrics"][key] for r in rows])) for key in rows[0]["metrics"]},
            "latency_mean_ms": float(np.mean([r["latency_ms"] for r in rows])),
            "latency_p95_ms": float(np.percentile([r["latency_ms"] for r in rows], 95))}


def evaluate(pipeline, samples, *, device="cpu", on_progress=None):
    if not samples:
        raise ValueError("No evaluation samples")
    retriever = pipeline.dense_retriever or pipeline.bm25_retriever
    ids = {p.passage_id for p in retriever.passages}
    if len({s["sample_id"] for s in samples}) != len(samples):
        raise ValueError("Duplicate sample IDs")
    for sample in samples:
        if not sample["query"].strip() or not sample["gold"]:
            raise ValueError(f"Empty query or gold: {sample['sample_id']}")
        if not set(sample["gold"]) <= ids:
            raise ValueError("Gold is missing from the index. Use the full corpus and matching segmentation; acceptance indexes cannot evaluate the frozen full-corpus set")
    def synchronize():
        if device.startswith("cuda"):
            import torch
            torch.cuda.synchronize(device)
    # Warm the actual service path; model load and warmup excluded from timing.
    pipeline.retrieve(RetrievalRequest(samples[0]["query"], top_k=10))
    rows = []
    for index, sample in enumerate(samples, 1):
        synchronize()
        started = time.perf_counter()
        evidence = pipeline.retrieve(RetrievalRequest(sample["query"], top_k=10))
        synchronize()
        latency = (time.perf_counter() - started) * 1000
        ranking = [e.passage_id for e in evidence]
        rows.append({"sample_id": sample["sample_id"], "domain": sample["domain"],
                     "gold": sample["gold"], "ranking": ranking,
                     "metrics": ranking_metrics(ranking, sample["gold"]), "latency_ms": latency})
        if on_progress:
            on_progress(index, len(samples))
    return {"summary": summarize(rows),
            "by_domain": {domain: summarize([r for r in rows if r["domain"] == domain])
                          for domain in sorted({r["domain"] for r in rows})},
            "samples": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation", choices=["section", "struct"], default="struct")
    parser.add_argument("--chain", choices=["rerank", "finetuned"], default="finetuned")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, default=ROOT / "artifacts/indexes")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--embedding-model")
    parser.add_argument("--reranker-model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all supplied samples")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    config = profile_config(segmentation=args.segmentation, chain=args.chain, index_root=args.index_root,
                            artifact_dir=args.artifact_dir, embedding_model=args.embedding_model,
                            reranker_model=args.reranker_model)
    samples = json.loads(args.samples.read_text(encoding="utf-8"))
    if args.limit:
        samples = samples[:args.limit]
    pipeline = HybridRetrievalPipeline.from_artifacts(artifact_dir=config.artifact_dir,
                                                       config=config, device=args.device)
    def progress(done, total):
        if done == 1 or done % 10 == 0 or done == total:
            print(f"Evaluated {done}/{total}", flush=True)
    result = evaluate(pipeline, samples, device=args.device, on_progress=progress)
    dense_metadata = json.loads((config.artifact_dir / "dense_metadata.json").read_text(encoding="utf-8"))
    result["configuration"] = {
        "segmentation": config.segmentation, "chain": args.chain,
        "embedding_model": config.embedding_model, "artifact_dir": str(config.artifact_dir),
        "device": args.device, "samples_file": str(args.samples), "limit": args.limit,
        "samples_file_sha256": hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        "evaluated_sample_ids_sha256": hashlib.sha256(json.dumps([s["sample_id"] for s in samples]).encode()).hexdigest(),
        "corpus_fingerprint": dense_metadata["passage_fingerprint"],
        "local_model_fingerprint": dense_metadata.get("local_model_fingerprint"),
        "query_contract": "frozen_input_unchanged_with_encoder_query_prefix",
        "gold_contract": "frozen_gold_passage_ids", "domain_filter": False,
        "gold_count_distribution": dict(Counter(len(set(s["gold"])) for s in samples)),
        "timing": "Warm serial service retrieval, includes thread pool and encoder inference; excludes loading, indexing, generation and evaluation LLM. Not directly comparable to the AutoDL timing benchmark.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved per-query rankings and domain summaries to {args.output}")


if __name__ == "__main__":
    main()
