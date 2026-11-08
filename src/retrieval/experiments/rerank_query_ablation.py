"""Isolated experiment: use the latest user turn only for Cross-Encoder scoring."""

import argparse
import json
import os
from pathlib import Path
import re

from retrieval.config import RetrievalConfig
from retrieval.evaluation import RetrievalEvalSample, evaluate_pipeline
from retrieval.pipeline import HybridRetrievalPipeline


class LatestUserReranker:
    """Change only the scoring query; preserve history-based candidate recall."""

    def __init__(self, reranker):
        self.reranker = reranker

    def rerank(self, main_query, sub_queries, candidates):
        user_turns = re.findall(r"^User:\s*(.+)$", main_query, flags=re.MULTILINE)
        current_query = user_turns[-1].strip() if user_turns else main_query
        return self.reranker.rerank(current_query, sub_queries, candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/acceptance"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/output/latest_user_rerank"))
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(Path(".model_cache").resolve()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    baseline = json.loads((args.baseline_dir / "evaluation_full.json").read_text())
    run = json.loads((args.baseline_dir / "summary.json").read_text())
    if run.get("status") != "passed" or not run.get("full"):
        raise ValueError("A passing full acceptance run is required")
    required_sample_fields = (
        "sample_id", "domain", "query",
        "user_gold_passage_ids", "agent_gold_passage_ids",
    )
    if any(key not in baseline["samples"][0] for key in required_sample_fields):
        raise ValueError("Baseline predates split user/agent Gold fields; rerun acceptance first")
    samples = [RetrievalEvalSample(**{
        key: row[key] for key in required_sample_fields
    }) for row in baseline["samples"]]
    pipeline = HybridRetrievalPipeline.from_artifacts(
        artifact_dir=args.artifact_dir,
        config=RetrievalConfig.for_mode("full", allow_reranker_fallback=False),
        device="cpu",
    )
    pipeline.reranker = LatestUserReranker(pipeline.reranker)
    measured = evaluate_pipeline(
        pipeline, samples,
        on_progress=lambda done, total: print(f"{done}/{total}", flush=True)
        if done % 10 == 0 else None,
    )
    measured["configuration"] = {
        "experiment": "latest_user_turn_for_reranker_only",
        "baseline_dir": str(args.baseline_dir),
        "candidate_queries": "unchanged full history",
        "corpus_fingerprint": run["corpus_fingerprint"],
        "sample_ids": [s.sample_id for s in samples],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(measured, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"baseline_full": baseline["summary"], "latest_user_only": measured["summary"]}, indent=2))


if __name__ == "__main__":
    main()
