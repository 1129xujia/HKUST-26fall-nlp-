"""Publish the team's native index assets for the main application CLI."""

import json
from pathlib import Path
import tempfile
import shutil

from .corpus import iter_passages, write_passages_jsonl, passage_fingerprint


def build_profile(config, documents, *, device="cpu", batch_size=16):
    """Publish an index only once every required artifact has been built."""
    target = Path(config.artifact_dir).resolve()
    if target.exists():
        raise FileExistsError(f"Index directory already exists: {target}; use a new --index-root/--artifact-dir to rebuild safely")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        passages = list(iter_passages(documents, segmentation=config.segmentation))
        write_passages_jsonl(passages, staging / "passages.jsonl")
        metadata = {"segmentation": config.segmentation, "passage_count": len(passages),
                    "passage_fingerprint": passage_fingerprint(passages)}
        (staging / "corpus_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if config.use_dense:
            from .dense import DenseRetriever

            DenseRetriever.build(
                passages, config.embedding_model, artifact_dir=staging, batch_size=batch_size,
                device=device, query_prefix=config.dense_query_prefix,
                passage_prefix=config.dense_passage_prefix,
            )
        if config.use_bm25:
            from .bm25 import BM25Retriever

            BM25Retriever.build(passages).save(staging)
        (staging / "service_manifest.json").write_text(json.dumps({
            **metadata, "chain": "finetuned" if config.mode == "finetuned" else "rerank",
            "embedding_model": config.embedding_model,
        }, indent=2), encoding="utf-8")
        staging.rename(target)
    finally:
        if staging.exists():
            if staging.resolve().parent != target.parent:
                raise RuntimeError("Temporary index directory escaped its intended parent")
            shutil.rmtree(staging)
    return target
