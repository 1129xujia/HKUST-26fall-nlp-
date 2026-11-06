"""FAISS-backed dense retrieval."""

import json
import sys
from pathlib import Path
from threading import Lock

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .corpus import PASSAGE_FINGERPRINT_VERSION, load_passages_jsonl, passage_fingerprint
from .exceptions import RetrievalIndexNotFound, RetrievalModelMismatch
from .schemas import Passage, SearchHit


class DenseRetriever:
    def __init__(self, passages, index, model, *, query_prefix="") -> None:
        self.passages, self.index, self.model = passages, index, model
        self.query_prefix = query_prefix
        self._encode_lock = Lock()
        # On macOS, this pinned Torch build can crash in CPU LayerNorm when
        # OpenMP runs inside a ThreadPoolExecutor worker. A single Torch CPU
        # thread avoids that native crash while BM25 and Dense remain parallel.
        if sys.platform == "darwin" and str(model.device).startswith("cpu"):
            import torch

            torch.set_num_threads(1)

    @classmethod
    def build(
        cls,
        passages: list[Passage],
        model_name: str,
        *,
        artifact_dir: Path | str,
        batch_size=64,
        device=None,
        query_prefix="",
        passage_prefix="",
        reuse_from: Path | str | None = None,
    ):
        if not passages:
            raise ValueError("cannot build dense index from an empty corpus")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model = SentenceTransformer(model_name, device=device)
        reused_count = 0
        if reuse_from is None:
            vectors = model.encode([passage_prefix + p.index_text for p in passages], batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        else:
            # 只有 passage ID 和完整 index_text 都一致才复用向量，避免行号与正文错配。
            reuse_from = Path(reuse_from)
            source_passages = load_passages_jsonl(reuse_from / "passages.jsonl")
            source_metadata = json.loads((reuse_from / "dense_metadata.json").read_text(encoding="utf-8"))
            source_vectors = np.load(reuse_from / "passage_embeddings.npy")
            if source_metadata.get("model_name") != model_name:
                raise RetrievalModelMismatch("Cannot reuse dense vectors from a different model")
            if source_metadata.get("passage_prefix", "") != passage_prefix:
                raise RetrievalModelMismatch("Cannot reuse dense vectors with a different passage prefix")
            if (
                source_metadata.get("passage_count") != len(source_passages)
                or source_metadata.get("passage_fingerprint") != passage_fingerprint(source_passages)
                or source_vectors.shape[0] != len(source_passages)
            ):
                raise RetrievalModelMismatch("Dense reuse source does not match its corpus")
            source_by_id = {
                passage.passage_id: (passage.index_text, source_vectors[index])
                for index, passage in enumerate(source_passages)
            }
            dimension = int(source_vectors.shape[1])
            vectors = np.empty((len(passages), dimension), dtype="float32")
            encode_rows: list[int] = []
            encode_texts: list[str] = []
            for row, passage in enumerate(passages):
                source = source_by_id.get(passage.passage_id)
                if source is not None and source[0] == passage.index_text:
                    vectors[row] = source[1]
                    reused_count += 1
                else:
                    encode_rows.append(row)
                    encode_texts.append(passage_prefix + passage.index_text)
            if encode_texts:
                encoded = model.encode(
                    encode_texts,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype("float32")
                vectors[np.asarray(encode_rows)] = encoded
        if not np.isfinite(vectors).all():
            raise RuntimeError("Embedding model returned non-finite passage vectors")
        # 向量已经归一化，因此内积等价于余弦相似度；IndexFlatIP 执行精确搜索。
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(artifact_dir / "faiss.index"))
        np.save(artifact_dir / "passage_embeddings.npy", vectors)
        metadata = {"model_name": model_name, "query_prefix": query_prefix, "passage_prefix": passage_prefix, "passage_count": len(passages), "dimension": int(vectors.shape[1]), "normalized": True, "fingerprint_version": PASSAGE_FINGERPRINT_VERSION, "passage_fingerprint": passage_fingerprint(passages), "reused_passage_count": reused_count, "encoded_passage_count": len(passages) - reused_count}
        (artifact_dir / "dense_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return cls(passages, index, model, query_prefix=query_prefix)

    @classmethod
    def from_artifacts(cls, artifact_dir: Path | str, model_name: str, *, passages=None, query_prefix="", passage_prefix="", device=None):
        artifact_dir = Path(artifact_dir)
        index_path, metadata_path = artifact_dir / "faiss.index", artifact_dir / "dense_metadata.json"
        if not index_path.exists() or not metadata_path.exists():
            raise RetrievalIndexNotFound(f"Missing dense artifacts in {artifact_dir}")
        passages = passages or load_passages_jsonl(artifact_dir / "passages.jsonl")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint_version") != PASSAGE_FINGERPRINT_VERSION:
            raise RetrievalModelMismatch("Dense fingerprint version is stale; rebuild indexes")
        if metadata.get("model_name") != model_name:
            raise RetrievalModelMismatch(f"Dense index uses {metadata.get('model_name')!r}, requested {model_name!r}")
        if metadata.get("passage_prefix", "") != passage_prefix:
            raise RetrievalModelMismatch("Dense index was built with a different passage prefix")
        if metadata.get("passage_count") != len(passages) or metadata.get("passage_fingerprint") != passage_fingerprint(passages):
            raise RetrievalModelMismatch("Dense index was built from a different corpus")
        index = faiss.read_index(str(index_path))
        if index.ntotal != len(passages) or index.d != metadata.get("dimension"):
            raise RetrievalModelMismatch("FAISS shape does not match corpus metadata")
        if metadata.get("normalized") is not True:
            raise RetrievalModelMismatch("Dense index must use normalized embeddings")
        return cls(passages, index, SentenceTransformer(model_name, device=device), query_prefix=query_prefix)

    def search(self, query: str, *, domain: str | None = None, top_k: int = 20) -> list[SearchHit]:
        if top_k <= 0:
            return []
        with self._encode_lock:
            vector = self.model.encode([self.query_prefix + query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        if not np.isfinite(vector).all():
            raise RuntimeError("Embedding model returned a non-finite query vector")
        search_k = len(self.passages) if domain else min(top_k, len(self.passages))
        scores, rows = self.index.search(vector, search_k)
        hits = []
        for score, row in zip(scores[0], rows[0]):
            if row < 0:
                continue
            item = self.passages[int(row)]
            if domain and item.domain != domain:
                continue
            hits.append(SearchHit(item, "dense", len(hits) + 1, float(score), query))
            if len(hits) >= top_k:
                break
        return hits
