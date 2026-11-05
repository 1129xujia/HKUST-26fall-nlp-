from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from customer_service_rag.data.io import load_passages, write_jsonl
from customer_service_rag.interfaces import EmbeddingModel
from customer_service_rag.schemas import Passage


def corpus_digest(passages: list[Passage]) -> str:
    content = "\n".join(p.model_dump_json() for p in passages)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass
class LocalIndex:
    passages: list[Passage]
    vectors: np.ndarray
    manifest: dict


def build_index(passages: list[Passage], embedding: EmbeddingModel, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    # An incomplete rebuild must never leave a valid-looking manifest.
    manifest_path.unlink(missing_ok=True)
    vectors = embedding.encode([f"{p.title}\n{p.text}" for p in passages])
    write_jsonl(directory / "passages.jsonl", passages)
    np.save(directory / "vectors.npy", vectors, allow_pickle=False)
    manifest = {
        "index_version": 1,
        "corpus_sha256": corpus_digest(passages),
        "vectors_sha256": hashlib.sha256((directory / "vectors.npy").read_bytes()).hexdigest(),
        "passage_count": len(passages),
        "embedding_model": embedding.model_name,
        "dimensions": embedding.dimensions,
        "is_mock_embedding": getattr(embedding, "backend", "hash") == "hash",
        "embedding_backend": getattr(embedding, "backend", "hash"),
        "bm25": "rebuilt from saved passages on load",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_index(directory: Path, embedding: EmbeddingModel) -> LocalIndex:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No completed index at {directory}; run cs-rag index first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    passages = load_passages(directory / "passages.jsonl")
    if manifest.get("index_version") != 1:
        raise ValueError("Unsupported index format; rebuild the index")
    if manifest["embedding_model"] != embedding.model_name:
        raise ValueError("Index embedding model differs from config; rebuild the index")
    if manifest["dimensions"] != embedding.dimensions:
        raise ValueError("Index embedding dimensions differ from config; rebuild the index")
    if manifest["corpus_sha256"] != corpus_digest(passages):
        raise ValueError("Saved passage corpus checksum mismatch; rebuild the index")
    vector_path = directory / "vectors.npy"
    if manifest["vectors_sha256"] != hashlib.sha256(vector_path.read_bytes()).hexdigest():
        raise ValueError("Vector checksum mismatch; rebuild the index")
    vectors = np.load(vector_path, allow_pickle=False)
    if vectors.shape != (len(passages), embedding.dimensions) or not np.isfinite(vectors).all():
        raise ValueError("Invalid vector shape or values; rebuild the index")
    if manifest["passage_count"] != len(passages):
        raise ValueError("Index passage count mismatch")
    return LocalIndex(passages=passages, vectors=vectors, manifest=manifest)
