"""Build an isolated GTE index and compare it with the current BGE baseline."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

from retrieval.experiments.optimize_retrieval import balanced_samples
from retrieval.bm25 import BM25Retriever
from retrieval.config import RetrievalConfig
from retrieval.corpus import load_passages_jsonl
from retrieval.dense import DenseRetriever
from retrieval.evaluation import build_validation_samples, evaluate_candidate_recall, evaluate_pipeline
from retrieval.pipeline import HybridRetrievalPipeline


E5_MODEL = "intfloat/e5-base-v2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--e5-artifact-dir", type=Path, default=Path("experiments/artifacts/e5-base-v2"))
    parser.add_argument("--output", type=Path, default=Path("experiments/output/embedding_model_comparison.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(Path(".model_cache").resolve()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    passages = load_passages_jsonl(args.artifact_dir / "passages.jsonl")
    dense_metadata = args.e5_artifact_dir / "dense_metadata.json"
    if dense_metadata.exists():
        e5 = DenseRetriever.from_artifacts(
            args.e5_artifact_dir, E5_MODEL, passages=passages,
            query_prefix="query: ", passage_prefix="passage: ", device=args.device,
        )
    else:
        e5 = DenseRetriever.build(
            passages, E5_MODEL, artifact_dir=args.e5_artifact_dir,
            batch_size=args.batch_size, device=args.device,
            query_prefix="query: ", passage_prefix="passage: ",
        )

    samples = balanced_samples(build_validation_samples(
        document_path=Path("multidoc2dial/multidoc2dial_doc.json"),
        dialogue_path=Path("multidoc2dial/multidoc2dial_dial_validation.json"),
        query_mode="history", history_turns=4,
    ), args.limit)
    bm25 = BM25Retriever.from_artifacts(args.artifact_dir, passages=passages)
    config = replace(
        RetrievalConfig.for_mode("rrf", artifact_dir=args.artifact_dir),
        embedding_model=E5_MODEL,
        dense_query_prefix="query: ",
        dense_passage_prefix="passage: ",
    )
    pipeline = HybridRetrievalPipeline(
        dense_retriever=e5,
        bm25_retriever=bm25,
        config=config,
    )
    dense_only = evaluate_candidate_recall(
        e5, samples, cutoffs=(10, 30, 50, 100, 200),
    )
    hybrid = evaluate_pipeline(pipeline, samples)
    baseline_path = args.artifact_dir / "evaluation_rrf_optimized_80.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None
    result = {
        "configuration": {
            "model": E5_MODEL,
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "sample_count": len(samples),
            "dense_top_k": config.dense_top_k,
            "bm25_top_k": config.bm25_top_k,
            "rrf_k": config.rrf_k,
            "reranker": False,
            "deduplication": False,
        },
        "bge_hybrid_baseline": None if baseline is None else baseline["summary"],
        "e5_dense_only": dense_only,
        "e5_bm25_hybrid": hybrid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "bge_hybrid_baseline": result["bge_hybrid_baseline"],
        "e5_dense_only": dense_only["summary"],
        "e5_bm25_hybrid": hybrid["summary"],
    }, indent=2), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
