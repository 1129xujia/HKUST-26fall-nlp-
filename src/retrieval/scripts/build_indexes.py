"""Build BM25 and dense artifacts from passages.jsonl."""

import argparse
from pathlib import Path

from retrieval.config import RetrievalConfig
from retrieval.corpus import load_passages_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--embedding-model", default=RetrievalConfig.embedding_model)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-bm25", action="store_true")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument(
        "--reuse-dense-from",
        type=Path,
        help="Reuse vectors only when passage ID and index text exactly match this artifact directory.",
    )
    args = parser.parse_args()
    if args.skip_bm25 and args.skip_dense:
        parser.error("cannot skip both indexes")

    passages = load_passages_jsonl(args.artifact_dir / "passages.jsonl")
    if not args.skip_bm25:
        from retrieval.bm25 import BM25Retriever

        BM25Retriever.build(passages).save(args.artifact_dir)
        print(f"Built BM25 index for {len(passages)} passages")
    if not args.skip_dense:
        from retrieval.dense import DenseRetriever

        DenseRetriever.build(
            passages,
            args.embedding_model,
            artifact_dir=args.artifact_dir,
            batch_size=args.batch_size,
            device=args.device,
            query_prefix=RetrievalConfig.dense_query_prefix,
            passage_prefix=RetrievalConfig.dense_passage_prefix,
            reuse_from=args.reuse_dense_from,
        )
        print(f"Built dense index for {len(passages)} passages")


if __name__ == "__main__":
    main()
