"""Diagnose and optimize Cross-Encoder reranking on a fixed RRF candidate pool."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

from retrieval.bm25 import BM25Retriever
from retrieval.config import RetrievalConfig
from retrieval.corpus import load_passages_jsonl
from retrieval.dense import DenseRetriever
from retrieval.evaluation import build_validation_samples
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import CrossEncoderReranker


QUERY_MODES = (
    "history",
    "current",
    "user_history",
    "current_first_user_history",
)


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


def passage_text(passage, mode):
    if mode == "index_text":
        return passage.index_text
    if mode == "labeled":
        parts = [
            f"Document: {passage.title}" if passage.title else "",
            f"Section: {passage.section_title}" if passage.section_title else "",
            f"Content: {passage.text}" if passage.text else "",
        ]
        return "\n".join(part for part in parts if part)
    if mode == "text_only":
        return passage.text
    raise ValueError(f"Unsupported passage mode: {mode}")


def cache_candidates(
    samples,
    dense,
    bm25,
    *,
    dense_depth,
    bm25_depth,
    candidate_depth,
    rrf_k,
):
    rows = []
    for done, sample in enumerate(samples, 1):
        dense_hits = dense.search(sample.query, domain=sample.domain, top_k=dense_depth)
        bm25_hits = bm25.search(sample.query, domain=sample.domain, top_k=bm25_depth)
        candidates = reciprocal_rank_fusion(
            [dense_hits, bm25_hits], rrf_k=rrf_k,
        )[:candidate_depth]
        rows.append({"sample": sample, "candidates": candidates})
        if done % 10 == 0 or done == len(samples):
            print(f"  candidates: {done}/{len(samples)}", flush=True)
    return rows


def score_candidates(
    rows,
    query_samples,
    reranker,
    *,
    passage_mode,
    cache_dir,
    cache_namespace,
    chunk_samples,
):
    offsets = []
    pair_count = 0
    for row, query_sample in zip(rows, query_samples, strict=True):
        start = pair_count
        pair_count += len(row["candidates"])
        offsets.append((start, pair_count))

    signature_payload = {
        "namespace": cache_namespace,
        "model": reranker.model_name,
        "passage_mode": passage_mode,
        "queries": [sample.query for sample in query_samples],
        "candidate_ids": [
            [candidate.passage.passage_id for candidate in row["candidates"]]
            for row in rows
        ],
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    run_cache = cache_dir / signature
    run_cache.mkdir(parents=True, exist_ok=True)
    score_chunks = []
    elapsed_ms = 0.0
    for sample_start in range(0, len(rows), chunk_samples):
        sample_end = min(sample_start + chunk_samples, len(rows))
        score_path = run_cache / f"{sample_start}_{sample_end}.npy"
        metadata_path = run_cache / f"{sample_start}_{sample_end}.json"
        expected = sum(len(row["candidates"]) for row in rows[sample_start:sample_end])
        chunk_scores = None
        if score_path.exists() and metadata_path.exists():
            loaded = np.load(score_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if loaded.shape == (expected,) and np.isfinite(loaded).all():
                chunk_scores = loaded
                elapsed_ms += float(metadata.get("elapsed_ms", 0.0))
                print(f"  score cache: samples {sample_start}:{sample_end}", flush=True)
        if chunk_scores is None:
            pairs = [
                (query_samples[index].query, passage_text(candidate.passage, passage_mode))
                for index in range(sample_start, sample_end)
                for candidate in rows[index]["candidates"]
            ]
            started = time.perf_counter()
            chunk_scores = np.asarray(reranker.model.predict(
                pairs,
                batch_size=reranker.batch_size,
                show_progress_bar=True,
            )).reshape(-1)
            chunk_elapsed_ms = (time.perf_counter() - started) * 1000
            if len(chunk_scores) != expected or not np.isfinite(chunk_scores).all():
                raise RuntimeError("Cross-Encoder returned invalid scores")
            np.save(score_path, chunk_scores)
            metadata_path.write_text(
                json.dumps({"elapsed_ms": chunk_elapsed_ms, "pair_count": expected}, indent=2),
                encoding="utf-8",
            )
            elapsed_ms += chunk_elapsed_ms
        score_chunks.append(chunk_scores)
    scores = np.concatenate(score_chunks) if score_chunks else np.empty(0)
    if len(scores) != pair_count or not np.isfinite(scores).all():
        raise RuntimeError("Cross-Encoder returned invalid scores")
    return scores, offsets, elapsed_ms


def summarize_depth(rows, scores, offsets, *, depth):
    cutoffs = (5, 10)
    hit = {cutoff: 0 for cutoff in cutoffs}
    recall = {cutoff: 0.0 for cutoff in cutoffs}
    user_hit = {cutoff: 0 for cutoff in cutoffs}
    user_recall = {cutoff: 0.0 for cutoff in cutoffs}
    agent_hit = {cutoff: 0 for cutoff in cutoffs}
    agent_recall = {cutoff: 0.0 for cutoff in cutoffs}
    user_count = sum(bool(row["sample"].user_gold_passage_ids) for row in rows)
    agent_count = sum(bool(row["sample"].agent_gold_passage_ids) for row in rows)
    reciprocal_ranks = []
    pre_hit = {cutoff: 0 for cutoff in cutoffs}
    pre_recall = {cutoff: 0.0 for cutoff in cutoffs}
    pre_reciprocal_ranks = []
    candidate_hits = promotions = demotions = unchanged = 0
    rescued_top10 = lost_top10 = rerank_hit10_given_candidate = 0
    details = []

    for row, (start, end) in zip(rows, offsets, strict=True):
        sample = row["sample"]
        candidates = row["candidates"][:depth]
        row_scores = scores[start:end][:len(candidates)]
        ranked_indexes = sorted(
            range(len(candidates)),
            key=lambda index: (
                -float(row_scores[index]),
                -candidates[index].rrf_score,
                candidates[index].passage.passage_id,
            ),
        )
        pre_ids = [candidate.passage.passage_id for candidate in candidates]
        post_ids = [candidates[index].passage.passage_id for index in ranked_indexes]
        gold = set(sample.gold_passage_ids)
        user_gold = set(sample.user_gold_passage_ids)
        agent_gold = set(sample.agent_gold_passage_ids)
        pre_best = next((rank for rank, pid in enumerate(pre_ids, 1) if pid in gold), None)
        post_best = next((rank for rank, pid in enumerate(post_ids, 1) if pid in gold), None)
        if pre_best is not None:
            candidate_hits += 1
            rerank_hit10_given_candidate += int(post_best is not None and post_best <= 10)
            promotions += int(post_best < pre_best)
            demotions += int(post_best > pre_best)
            unchanged += int(post_best == pre_best)
            rescued_top10 += int(pre_best > 10 and post_best <= 10)
            lost_top10 += int(pre_best <= 10 and post_best > 10)
        reciprocal_ranks.append(
            0.0 if post_best is None or post_best > 10 else 1.0 / post_best
        )
        pre_reciprocal_ranks.append(
            0.0 if pre_best is None or pre_best > 10 else 1.0 / pre_best
        )
        for cutoff in cutoffs:
            pre_retrieved = set(pre_ids[:cutoff])
            pre_overlap = pre_retrieved & gold
            pre_hit[cutoff] += int(bool(pre_overlap))
            pre_recall[cutoff] += len(pre_overlap) / len(gold)
            retrieved = set(post_ids[:cutoff])
            overlap = retrieved & gold
            hit[cutoff] += int(bool(overlap))
            recall[cutoff] += len(overlap) / len(gold)
            if user_gold:
                user_overlap = retrieved & user_gold
                user_hit[cutoff] += int(bool(user_overlap))
                user_recall[cutoff] += len(user_overlap) / len(user_gold)
            if agent_gold:
                agent_overlap = retrieved & agent_gold
                agent_hit[cutoff] += int(bool(agent_overlap))
                agent_recall[cutoff] += len(agent_overlap) / len(agent_gold)
        details.append({
            "sample_id": sample.sample_id,
            "candidate_has_gold": pre_best is not None,
            "gold_rank_before": pre_best,
            "gold_rank_after": post_best,
            "rank_improvement": None if pre_best is None else pre_best - post_best,
            "pre_top10_ids": pre_ids[:10],
            "post_top10_ids": post_ids[:10],
        })

    count = len(rows)
    summary = {
        "sample_count": count,
        "candidate_depth": depth,
        "candidate_hit_rate": candidate_hits / count,
        "pre_rerank_mrr@10": statistics.fmean(pre_reciprocal_ranks),
        "mrr@10": statistics.fmean(reciprocal_ranks),
        "promotion_rate_given_candidate": promotions / candidate_hits if candidate_hits else 0.0,
        "demotion_rate_given_candidate": demotions / candidate_hits if candidate_hits else 0.0,
        "unchanged_rate_given_candidate": unchanged / candidate_hits if candidate_hits else 0.0,
        "rerank_hit_rate@10_given_candidate": (
            rerank_hit10_given_candidate / candidate_hits if candidate_hits else 0.0
        ),
        "rescued_to_top10_count": rescued_top10,
        "lost_from_top10_count": lost_top10,
    }
    for cutoff in cutoffs:
        summary[f"pre_rerank_hit_rate@{cutoff}"] = pre_hit[cutoff] / count
        summary[f"pre_rerank_recall@{cutoff}"] = pre_recall[cutoff] / count
        summary[f"hit_rate@{cutoff}"] = hit[cutoff] / count
        summary[f"recall@{cutoff}"] = recall[cutoff] / count
        summary[f"user_gold_hit_rate@{cutoff}"] = user_hit[cutoff] / user_count
        summary[f"user_gold_recall@{cutoff}"] = user_recall[cutoff] / user_count
        summary[f"agent_gold_hit_rate@{cutoff}"] = agent_hit[cutoff] / agent_count
        summary[f"agent_gold_recall@{cutoff}"] = agent_recall[cutoff] / agent_count
    return {"summary": summary, "samples": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--depths", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--dense-depth", type=int)
    parser.add_argument("--bm25-depth", type=int)
    parser.add_argument("--query-modes", nargs="+", choices=QUERY_MODES, default=list(QUERY_MODES))
    parser.add_argument("--passage-mode", choices=["index_text", "labeled", "text_only"], default="index_text")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("experiments/output/reranker_optimization.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--score-cache-dir", type=Path, default=Path("experiments/cache/reranker_scores"))
    parser.add_argument("--score-chunk-samples", type=int, default=10)
    args = parser.parse_args()
    if (
        args.limit <= 0
        or args.offset < 0
        or args.score_chunk_samples <= 0
        or (args.dense_depth is not None and args.dense_depth <= 0)
        or (args.bm25_depth is not None and args.bm25_depth <= 0)
        or any(depth <= 0 for depth in args.depths)
    ):
        parser.error("limit/depth/chunk must be positive and offset non-negative")
    os.environ.setdefault("HF_HOME", str(Path(".model_cache").resolve()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    document_path = Path("multidoc2dial/multidoc2dial_doc.json")
    dialogue_path = Path("multidoc2dial/multidoc2dial_dial_validation.json")
    history = balanced_samples(build_validation_samples(
        document_path=document_path,
        dialogue_path=dialogue_path,
        query_mode="history",
        history_turns=4,
    ), args.limit + args.offset)[args.offset:]
    samples_by_mode = {"history": history}
    for mode in set(args.query_modes) - {"history"}:
        samples_by_mode[mode] = align_samples(history, build_validation_samples(
            document_path=document_path,
            dialogue_path=dialogue_path,
            query_mode=mode,
            history_turns=4,
        ))

    config = RetrievalConfig()
    passages = load_passages_jsonl(args.artifact_dir / "passages.jsonl")
    dense = DenseRetriever.from_artifacts(
        args.artifact_dir,
        config.embedding_model,
        passages=passages,
        query_prefix=config.dense_query_prefix,
        passage_prefix=config.dense_passage_prefix,
        device=args.device,
    )
    bm25 = BM25Retriever.from_artifacts(args.artifact_dir, passages=passages)
    max_depth = max(args.depths)
    dense_depth = args.dense_depth or max_depth
    bm25_depth = args.bm25_depth or max_depth
    print(f"Caching fixed RRF Top {max_depth} candidates...", flush=True)
    rows = cache_candidates(
        history,
        dense,
        bm25,
        dense_depth=dense_depth,
        bm25_depth=bm25_depth,
        candidate_depth=max_depth,
        rrf_k=config.rrf_k,
    )
    reranker = CrossEncoderReranker(
        config.reranker_model,
        batch_size=args.batch_size,
        device=args.device,
    )

    results = {}
    for mode in args.query_modes:
        print(f"Scoring query mode: {mode}", flush=True)
        scores, offsets, elapsed_ms = score_candidates(
            rows,
            samples_by_mode[mode],
            reranker,
            passage_mode=args.passage_mode,
            cache_dir=args.score_cache_dir,
            cache_namespace=(
                f"offset={args.offset};dense={dense_depth};bm25={bm25_depth};"
                f"candidate={max_depth};query={mode}"
            ),
            chunk_samples=args.score_chunk_samples,
        )
        by_depth = {}
        for depth in sorted(set(args.depths)):
            measured = summarize_depth(rows, scores, offsets, depth=depth)
            measured["summary"]["score_total_ms_at_max_depth"] = elapsed_ms
            measured["summary"]["mean_score_ms_per_sample_at_max_depth"] = elapsed_ms / len(rows)
            by_depth[str(depth)] = measured
            summary = measured["summary"]
            print(
                f"  depth={depth} hit@10={summary['hit_rate@10']:.4f} "
                f"recall@10={summary['recall@10']:.4f} mrr@10={summary['mrr@10']:.4f} "
                f"promote={summary['promotion_rate_given_candidate']:.4f} "
                f"demote={summary['demotion_rate_given_candidate']:.4f}",
                flush=True,
            )
        results[mode] = {
            "scored_pair_count": len(scores),
            "score_total_ms": elapsed_ms,
            "depths": by_depth,
        }

    payload = {
        "configuration": {
            "sample_count": len(history),
            "sample_offset": args.offset,
            "sample_ids": [sample.sample_id for sample in history],
            "candidate_query_mode": "history",
            "rerank_query_modes": args.query_modes,
            "passage_mode": args.passage_mode,
            "branch_depth": max_depth,
            "dense_depth": dense_depth,
            "bm25_depth": bm25_depth,
            "depths": sorted(set(args.depths)),
            "rrf_k": config.rrf_k,
            "reranker_model": config.reranker_model,
            "batch_size": args.batch_size,
            "domain_filter": True,
            "extra_deduplication": False,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
