"""Run one retrieval request and print JSON evidence."""

import argparse
import json
import sys
from pathlib import Path

from retrieval.config import RetrievalConfig
from retrieval.pipeline import HybridRetrievalPipeline
from retrieval.schemas import RetrievalRequest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--sub-query", action="append", default=[])
    parser.add_argument("--domain", choices=["dmv", "ssa", "va", "studentaid"])
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--mode", "--method", dest="mode", choices=["bm25", "dense", "rrf", "full", "finetuned", "rerank"], default="rrf")
    parser.add_argument("--segmentation", choices=["section", "struct"], default="section")
    parser.add_argument("--embedding-model")
    parser.add_argument("--reranker-model")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = RetrievalConfig.for_mode(args.mode, artifact_dir=args.artifact_dir,
                                      segmentation=args.segmentation,
                                      embedding_model=args.embedding_model,
                                      reranker_model=args.reranker_model,
                                      allow_reranker_fallback=False)

    pipeline = HybridRetrievalPipeline.from_artifacts(
        artifact_dir=args.artifact_dir, config=config, device=args.device
    )
    evidence = pipeline.retrieve(
        RetrievalRequest(
            query=args.query,
            sub_queries=args.sub_query,
            domain=args.domain,
            top_k=args.top_k,
        )
    )
    print(json.dumps([item.to_dict() for item in evidence], indent=2, ensure_ascii=False))
    if pipeline.runtime_notes:
        print(json.dumps({"runtime_notes": pipeline.runtime_notes}, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
