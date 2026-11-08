"""Build a Dense index in resumable chunks for constrained local runtimes."""

import argparse
import json
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval.corpus import PASSAGE_FINGERPRINT_VERSION, load_passages_jsonl, passage_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--passages", type=Path, default=Path("artifacts/passages.jsonl"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--passage-prefix", default="")
    parser.add_argument("--chunk-size", type=int, default=1500)
    parser.add_argument("--chunk-index", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(Path(".model_cache").resolve()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    passages = load_passages_jsonl(args.passages)
    chunks_dir = args.artifact_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    if args.finalize:
        files = sorted(chunks_dir.glob("*.npy"), key=lambda path: int(path.stem.split("_")[0]))
        expected_start = 0
        arrays = []
        for path in files:
            start, end = map(int, path.stem.split("_"))
            if start != expected_start:
                raise RuntimeError(f"Missing chunk before {path}; expected start {expected_start}")
            array = np.load(path)
            if array.ndim != 2 or len(array) != end - start or not np.isfinite(array).all():
                raise RuntimeError(f"Invalid embedding chunk: {path}")
            arrays.append(array)
            expected_start = end
        if expected_start != len(passages):
            raise RuntimeError(f"Chunks cover {expected_start}/{len(passages)} passages")
        vectors = np.concatenate(arrays).astype("float32", copy=False)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(args.artifact_dir / "faiss.index"))
        np.save(args.artifact_dir / "passage_embeddings.npy", vectors)
        metadata = {
            "model_name": args.model,
            "query_prefix": args.query_prefix,
            "passage_prefix": args.passage_prefix,
            "passage_count": len(passages),
            "dimension": int(vectors.shape[1]),
            "normalized": True,
            "fingerprint_version": PASSAGE_FINGERPRINT_VERSION,
            "passage_fingerprint": passage_fingerprint(passages),
            "build_method": "resumable_chunks",
        }
        (args.artifact_dir / "dense_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(f"Finalized {len(passages)} vectors in {args.artifact_dir}")
        return

    if args.chunk_index is None or args.chunk_index < 0:
        parser.error("--chunk-index must be a non-negative integer unless --finalize is used")
    start = args.chunk_index * args.chunk_size
    end = min(start + args.chunk_size, len(passages))
    if start >= len(passages):
        print(f"Chunk {args.chunk_index} is beyond corpus end; nothing to do")
        return
    output = chunks_dir / f"{start}_{end}.npy"
    if output.exists():
        existing = np.load(output, mmap_mode="r")
        if existing.ndim == 2 and len(existing) == end - start:
            print(f"Chunk already valid: {output}")
            return
        raise RuntimeError(f"Existing chunk is invalid: {output}")
    model = SentenceTransformer(args.model, device=args.device)
    vectors = model.encode(
        [args.passage_prefix + passage.index_text for passage in passages[start:end]],
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    if not np.isfinite(vectors).all():
        raise RuntimeError("Embedding chunk contains non-finite values")
    np.save(output, vectors)
    print(f"Wrote chunk {args.chunk_index}: {start}:{end} -> {output}")


if __name__ == "__main__":
    main()
