"""Build or query an independent section/struct × rerank/finetuned profile."""

import argparse
import json
from pathlib import Path
import tempfile
import shutil

from .profiles import profile_config
from .corpus import iter_passages, write_passages_jsonl, passage_fingerprint
from .pipeline import HybridRetrievalPipeline
from .schemas import RetrievalRequest


def build_profile(config, documents, *, device="cpu", batch_size=16):
    """Publish an index only once every required artifact has been built."""
    target = config.artifact_dir
    if target.exists():
        raise FileExistsError(f"Index directory already exists: {target}; use a new --index-root/--artifact-dir to rebuild safely")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        from .dense import DenseRetriever
        from .bm25 import BM25Retriever

        passages = list(iter_passages(documents, segmentation=config.segmentation))
        write_passages_jsonl(passages, staging / "passages.jsonl")
        metadata = {"segmentation": config.segmentation, "passage_count": len(passages),
                    "passage_fingerprint": passage_fingerprint(passages)}
        (staging / "corpus_metadata.json").write_text(json.dumps(metadata, indent=2))
        DenseRetriever.build(
            passages, config.embedding_model, artifact_dir=staging, batch_size=batch_size,
            device=device, query_prefix=config.dense_query_prefix,
            passage_prefix=config.dense_passage_prefix,
        )
        if config.use_bm25:
            BM25Retriever.build(passages).save(staging)
        (staging / "service_manifest.json").write_text(json.dumps({
            **metadata, "chain": "finetuned" if config.mode == "finetuned" else "rerank",
            "embedding_model": config.embedding_model,
        }, indent=2))
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "search"])
    parser.add_argument("--segmentation", choices=["section", "struct"], required=True)
    parser.add_argument("--chain", choices=["rerank", "finetuned"], required=True)
    parser.add_argument("--index-root", type=Path, default=Path("artifacts/indexes"))
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--embedding-model")
    parser.add_argument("--reranker-model")
    parser.add_argument("--documents", type=Path, default=Path("multidoc2dial/multidoc2dial_doc.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--domain", choices=["dmv", "ssa", "studentaid", "va"])
    args = parser.parse_args()
    config = profile_config(segmentation=args.segmentation, chain=args.chain,
                            index_root=args.index_root, artifact_dir=args.artifact_dir,
                            embedding_model=args.embedding_model, reranker_model=args.reranker_model)
    if args.action == "build":
        print(build_profile(config, args.documents, device=args.device, batch_size=args.batch_size))
        return
    if not args.query:
        parser.error("search requires --query")
    pipeline = HybridRetrievalPipeline.from_artifacts(
        artifact_dir=config.artifact_dir, config=config, device=args.device)
    evidence = pipeline.retrieve(RetrievalRequest(args.query, domain=args.domain, top_k=args.top_k))
    print(json.dumps({"segmentation": config.segmentation, "chain": args.chain,
                      "embedding_model": config.embedding_model,
                      "evidence": [e.to_dict() for e in evidence]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
