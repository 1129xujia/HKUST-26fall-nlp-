"""Measure BM25, Dense, or fused-RRF candidate recall before reranking."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

from retrieval.bm25 import BM25Retriever
from retrieval.config import RetrievalConfig
from retrieval.corpus import load_passages_jsonl
from retrieval.dense import DenseRetriever
from retrieval.evaluation import build_validation_samples, evaluate_candidate_recall
from retrieval.fusion import reciprocal_rank_fusion


class RrfCandidateRetriever:
    """Expose the fused pre-rerank candidate list through the search interface."""

    def __init__(self, dense, bm25, *, rrf_k: int):
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k

    def search(self, query, *, domain=None, top_k=20):
        ranked_lists = [
            self.dense.search(query, domain=domain, top_k=top_k),
            self.bm25.search(query, domain=domain, top_k=top_k),
        ]
        return reciprocal_rank_fusion(ranked_lists, rrf_k=self.rrf_k)[:top_k]


def balanced_samples(samples, limit):
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.domain].append(sample)
    chosen = []
    for row in range(max(map(len, groups.values()), default=0)):
        for domain in sorted(groups):
            if row < len(groups[domain]):
                chosen.append(groups[domain][row])
                if limit is not None and len(chosen) == limit:
                    return chosen
    return chosen


def print_cutoff_table(name, summary, cutoffs):
    print(f"\n{name.upper()} pre-rerank candidate coverage", flush=True)
    print("cutoff  hit_rate  recall  user_recall  agent_recall", flush=True)
    for cutoff in cutoffs:
        print(
            f"@{cutoff:<5} "
            f"{summary[f'hit_rate@{cutoff}'] * 100:>7.2f}% "
            f"{summary[f'recall@{cutoff}'] * 100:>7.2f}% "
            f"{summary[f'user_gold_recall@{cutoff}'] * 100:>11.2f}% "
            f"{summary[f'agent_gold_recall@{cutoff}'] * 100:>12.2f}%",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes", nargs="+", choices=["bm25", "dense", "rrf"],
        default=["bm25", "dense", "rrf"],
    )
    parser.add_argument(
        "--cutoffs", nargs="+", type=int, default=[5, 10, 50, 100, 200]
    )
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--documents", type=Path, default=Path("multidoc2dial/multidoc2dial_doc.json"))
    parser.add_argument("--dialogues", type=Path, default=Path("multidoc2dial/multidoc2dial_dial_validation.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--global-search", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/candidate_recall.json"))
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    os.environ.setdefault("HF_HOME", str(Path(".model_cache").resolve()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = RetrievalConfig()
    passages = load_passages_jsonl(args.artifact_dir / "passages.jsonl")
    samples = balanced_samples(build_validation_samples(
        document_path=args.documents,
        dialogue_path=args.dialogues,
        query_mode="history",
        history_turns=4,
    ), args.limit)

    retrievers = {}
    bm25 = None
    dense = None
    if "bm25" in args.modes or "rrf" in args.modes:
        bm25 = BM25Retriever.from_artifacts(
            args.artifact_dir, passages=passages,
        )
    if "dense" in args.modes or "rrf" in args.modes:
        dense = DenseRetriever.from_artifacts(
            args.artifact_dir,
            config.embedding_model,
            passages=passages,
            query_prefix=config.dense_query_prefix,
            device=args.device,
        )
    if "bm25" in args.modes:
        retrievers["bm25"] = bm25
    if "dense" in args.modes:
        retrievers["dense"] = dense
    if "rrf" in args.modes:
        retrievers["rrf"] = RrfCandidateRetriever(
            dense, bm25, rrf_k=config.rrf_k,
        )

    results = {}
    for mode, retriever in retrievers.items():
        print(f"Evaluating raw {mode} candidate recall on {len(samples)} samples...", flush=True)
        results[mode] = evaluate_candidate_recall(
            retriever,
            samples,
            cutoffs=tuple(args.cutoffs),
            use_domain_filter=not args.global_search,
            on_progress=lambda done, total, name=mode: print(
                f"  {name}: {done}/{total}", flush=True
            ) if done % 10 == 0 or done == total else None,
        )
        print_cutoff_table(
            mode, results[mode]["summary"], sorted(set(args.cutoffs))
        )

    payload = {
        "configuration": {
            "modes": args.modes,
            "cutoffs": sorted(set(args.cutoffs)),
            "sample_count": len(samples),
            "query_mode": "history",
            "history_turns": 4,
            "rrf_k": config.rrf_k,
            "rrf_branch_depth": max(args.cutoffs),
            "stage": "before_reranker",
            "domain_filter": not args.global_search,
            "sample_strategy": "deterministic_round_robin_by_domain",
            "sample_ids": [sample.sample_id for sample in samples],
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote candidate recall diagnostics to {args.output}", flush=True)


if __name__ == "__main__":
    main()
