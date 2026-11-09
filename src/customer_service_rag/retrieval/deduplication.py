"""Exact and embedding-aware evidence de-duplication."""

import json
import re
from pathlib import Path

import numpy as np

from .corpus import PASSAGE_FINGERPRINT_VERSION, passage_fingerprint
from .exceptions import RetrievalModelMismatch


def _normalise(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


class NoOpDeduplicator:
    """Preserve ranking and only apply the requested final Top-K cutoff."""

    def select(self, candidates, *, top_k: int):
        return [] if top_k <= 0 else list(candidates[:top_k])


class EvidenceDeduplicator:
    def __init__(self, passages=None, embeddings=None, *, max_passages_per_doc=2, similarity_threshold=0.93):
        self.embeddings = embeddings
        self.max_passages_per_doc = max_passages_per_doc
        self.similarity_threshold = similarity_threshold
        self._row_by_id = {item.passage_id: row for row, item in enumerate(passages or [])}

    @classmethod
    def from_artifacts(cls, *, artifact_dir: Path | str, passages, max_passages_per_doc=2, similarity_threshold=0.93):
        artifact_dir = Path(artifact_dir)
        vector_path = artifact_dir / "passage_embeddings.npy"
        if not vector_path.exists():
            return cls(passages, max_passages_per_doc=max_passages_per_doc, similarity_threshold=similarity_threshold)
        metadata_path = artifact_dir / "dense_metadata.json"
        if not metadata_path.exists():
            raise RetrievalModelMismatch("Dense metadata is required with passage embeddings")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint_version") != PASSAGE_FINGERPRINT_VERSION:
            raise RetrievalModelMismatch("Embedding fingerprint version is stale")
        if metadata.get("passage_count") != len(passages) or metadata.get("passage_fingerprint") != passage_fingerprint(passages):
            raise RetrievalModelMismatch("Embeddings were built from a different corpus")
        vectors = np.load(vector_path, mmap_mode="r")
        if vectors.ndim != 2 or vectors.shape != (len(passages), metadata.get("dimension")):
            raise RetrievalModelMismatch("Embedding shape does not match corpus metadata")
        if metadata.get("normalized") is not True:
            raise RetrievalModelMismatch("Semantic deduplication requires normalized embeddings")
        return cls(passages, vectors, max_passages_per_doc=max_passages_per_doc, similarity_threshold=similarity_threshold)

    def select(self, candidates, *, top_k: int):
        if top_k <= 0:
            return []
        selected, selected_rows = [], []
        seen_ids, seen_text, per_doc = set(), set(), {}
        for candidate in candidates:
            item = candidate.passage
            normalized = _normalise(item.text)
            if item.passage_id in seen_ids or normalized in seen_text:
                continue
            if per_doc.get(item.doc_id, 0) >= self.max_passages_per_doc:
                continue
            row = self._row_by_id.get(item.passage_id)
            if self.embeddings is not None and row is not None and any(float(np.dot(np.asarray(self.embeddings[row]), np.asarray(self.embeddings[other]))) >= self.similarity_threshold for other in selected_rows):
                continue
            selected.append(candidate)
            seen_ids.add(item.passage_id)
            seen_text.add(normalized)
            per_doc[item.doc_id] = per_doc.get(item.doc_id, 0) + 1
            if row is not None:
                selected_rows.append(row)
            if len(selected) >= top_k:
                break
        return selected
