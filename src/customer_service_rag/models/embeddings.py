"""Deterministic lexical hash vectors, NOT learned semantic embeddings."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from customer_service_rag.config import EmbeddingConfig


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[\u4e00-\u9fff]", text.casefold())


class HashEmbedding:
    model_name = "hash-token-v1"
    backend = "hash"

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 8:
            raise ValueError("Embedding dimensions must be at least 8")
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                column = int.from_bytes(digest[:8], "little") % self.dimensions
                vectors[row, column] += 1.0 if digest[8] % 2 == 0 else -1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)


class SemanticEmbedding:
    backend = "transformers"

    def __init__(self, config: EmbeddingConfig, root: Path) -> None:
        from customer_service_rag.models.runtime import load_runtime
        path = config.local_path
        path = path if path.is_absolute() else root / path
        self.config = config
        self.dimensions = config.dimensions
        self.model_name = f"{config.model}@{config.revision};mean;max_length={config.max_length}"
        self.runtime = load_runtime(
            str(path.resolve()), "embedding", config.model, config.revision, config.num_threads,
        )
        if self.runtime.model.config.hidden_size != config.dimensions:
            raise ValueError("Configured embedding dimensions do not match the model")

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        runtime = self.runtime
        with runtime.lock, runtime.torch.inference_mode():
            for start in range(0, len(texts), self.config.batch_size):
                inputs = runtime.tokenizer(
                    texts[start:start + self.config.batch_size], padding=True,
                    truncation=True, max_length=self.config.max_length, return_tensors="pt",
                )
                hidden = runtime.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled = runtime.torch.nn.functional.normalize(pooled, p=2, dim=1)
                rows.append(pooled.cpu().numpy())
        return np.concatenate(rows, axis=0) if rows else np.empty((0, self.dimensions), np.float32)


def build_embedding(config: EmbeddingConfig, root: Path | None = None):
    if config.backend == "transformers":
        return SemanticEmbedding(config, root or Path.cwd())
    return HashEmbedding(dimensions=config.dimensions)
