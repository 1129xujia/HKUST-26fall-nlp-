"""Cached ablations for deduplication, query construction, candidate depth, and RRF."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics

from retrieval.bm25 import BM25Retriever
from retrieval.config import RetrievalConfig
from retrieval.corpus import load_passages_jsonl
from retrieval.deduplication import EvidenceDeduplicator
from retrieval.dense import DenseRetriever
from retrieval.evaluation import RetrievalEvalSample, build_validation_samples
from retrieval.fusion import reciprocal_rank_fusion


def balanced_samples(samples, limit):
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.domain].append(sample)
    chosen = []
    for row in range(max(map(len, groups.values()), default=0)):
        for domain in sorted(groups):
            if row < len(groups[domain]):
                chosen.append(groups[domain][row])
                if len(chosen) == limit:
                    return chosen
    return chosen


def align_samples(reference, candidates):
    by_id = {sample.sample_id: sample for sample in candidates}
    return [by_id[sample.sample_id] for sample in reference]


def cache_rankings(samples, dense, bm25, max_k):
    cached = []
    for index, sample in enumerate(samples, 1):
        cached.append({
            "sample": sample,
            "dense": dense.search(sample.query, domain=sample.domain, top_k=max_k),
            "bm25": bm25.search(sample.query, domain=sample.domain, top_k=max_k),
        })
        if index % 10 == 0 or index == len(samples):
            print(f"  cached {index}/{len(samples)}", flush=True)
    return cached


def summarize(rows, selected_ids):
    cutoffs = (5, 10)
    groups = ("all", "user", "agent")
    counts = {
        "all": len(rows),
        "user": sum(bool(row["sample"].user_gold_passage_ids) for row in rows),
        "agent": sum(bool(row["sample"].agent_gold_passage_ids) for row in rows),
    }
    hit = {group: {k: 0 for k in cutoffs} for group in groups}
    recall = {group: {k: 0.0 for k in cutoffs} for group in groups}
    reciprocal_ranks = []
    result_counts = []
    for row, ids in zip(rows, selected_ids, strict=True):
        sample = row["sample"]
        golds = {
            "all": set(sample.gold_passage_ids),
            "user": set(sample.user_gold_passage_ids),
            "agent": set(sample.agent_gold_passage_ids),
        }
        result_counts.append(len(ids))
        for cutoff in cutoffs:
            retrieved = set(ids[:cutoff])
            for group, gold in golds.items():
                if not gold:
                    continue
                overlap = retrieved & gold
                hit[group][cutoff] += int(bool(overlap))
                recall[group][cutoff] += len(overlap) / len(gold)
        first = next((rank for rank, pid in enumerate(ids, 1) if pid in golds["all"]), None)
        reciprocal_ranks.append(0.0 if first is None else 1.0 / first)

    summary = {
        "sample_count": len(rows),
        "mrr@10": statistics.fmean(reciprocal_ranks),
        "mean_result_count": statistics.fmean(result_counts),
    }
    for cutoff in cutoffs:
        for group in groups:
            prefix = "" if group == "all" else f"{group}_gold_"
            denominator = counts[group]
            summary[f"{prefix}hit_rate@{cutoff}"] = hit[group][cutoff] / denominator
            summary[f"{prefix}recall@{cutoff}"] = recall[group][cutoff] / denominator
    return summary


def run_configuration(rows, *, dense_k, bm25_k, rrf_k, fused_k, final_k, selector=None):
    selected_ids = []
    for row in rows:
        fused = reciprocal_rank_fusion(
            [row["dense"][:dense_k], row["bm25"][:bm25_k]],
            rrf_k=rrf_k,
        )[:fused_k]
        selected = fused[:final_k] if selector is None else selector.select(fused, top_k=final_k)
        selected_ids.append([candidate.passage.passage_id for candidate in selected])
    return summarize(rows, selected_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("experiments/output/retrieval_optimization.json"))
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    os.environ.setdefault("HF_HOME", str(Path(".model_cache").resolve()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    documents = Path("multidoc2dial/multidoc2dial_doc.json")
    dialogues = Path("multidoc2dial/multidoc2dial_dial_validation.json")
    history = balanced_samples(build_validation_samples(
        document_path=documents, dialogue_path=dialogues,
        query_mode="history", history_turns=4,
    ), args.limit + args.offset)[args.offset:]
    samples_by_mode = {"history": history}
    for query_mode in ("current", "user_history", "current_first_user_history"):
        samples_by_mode[query_mode] = align_samples(history, build_validation_samples(
            document_path=documents, dialogue_path=dialogues,
            query_mode=query_mode, history_turns=4,
        ))
    passages = load_passages_jsonl(args.artifact_dir / "passages.jsonl")
    config = RetrievalConfig()
    bm25 = BM25Retriever.from_artifacts(args.artifact_dir, passages=passages)
    dense = DenseRetriever.from_artifacts(
        args.artifact_dir, config.embedding_model, passages=passages,
        query_prefix=config.dense_query_prefix, device=args.device,
    )

    caches = {}
    for name, samples in samples_by_mode.items():
        print(f"Caching {name} rankings at Top 200...", flush=True)
        caches[name] = cache_rankings(samples, dense, bm25, 200)

    full_dedup = EvidenceDeduplicator.from_artifacts(
        artifact_dir=args.artifact_dir, passages=passages,
        max_passages_per_doc=2, similarity_threshold=0.93,
    )
    selectors = {
        "off": None,
        "exact_text_only": EvidenceDeduplicator(
            passages, max_passages_per_doc=len(passages), similarity_threshold=1.0,
        ),
        "doc_limit_only": EvidenceDeduplicator(
            passages, max_passages_per_doc=2, similarity_threshold=1.0,
        ),
        "semantic_only": EvidenceDeduplicator(
            passages, full_dedup.embeddings,
            max_passages_per_doc=len(passages), similarity_threshold=0.93,
        ),
        "full": full_dedup,
    }
    dedup_ablation = {}
    for name, selector in selectors.items():
        dedup_ablation[name] = run_configuration(
            caches["history"], dense_k=30, bm25_k=30, rrf_k=60,
            fused_k=40, final_k=10, selector=selector,
        )

    query_ablation = {}
    for name, rows in caches.items():
        query_ablation[name] = run_configuration(
            rows, dense_k=30, bm25_k=30, rrf_k=60,
            fused_k=40, final_k=10, selector=None,
        )

    grid = []
    for query_mode, rows in caches.items():
        for dense_k in (30, 50, 100, 200):
            for bm25_k in (30, 50, 100, 200):
                for rrf_k in (10, 30, 60, 100):
                    summary = run_configuration(
                        rows, dense_k=dense_k, bm25_k=bm25_k,
                        rrf_k=rrf_k, fused_k=400, final_k=10, selector=None,
                    )
                    grid.append({
                        "query_mode": query_mode,
                        "dense_k": dense_k,
                        "bm25_k": bm25_k,
                        "rrf_k": rrf_k,
                        **summary,
                    })
    grid.sort(key=lambda row: (row["recall@10"], row["hit_rate@10"], row["mrr@10"]), reverse=True)

    output = {
        "configuration": {
            "sample_count": args.limit,
            "sample_offset": args.offset,
            "sample_ids": [sample.sample_id for sample in history],
            "domain_filter": True,
            "reranker": False,
            "final_k": 10,
        },
        "dedup_ablation": dedup_ablation,
        "query_ablation": query_ablation,
        "candidate_rrf_grid": grid,
        "best_candidate_rrf": grid[0],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "dedup_ablation": dedup_ablation,
        "query_ablation": query_ablation,
        "best_candidate_rrf": grid[0],
    }, indent=2), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
