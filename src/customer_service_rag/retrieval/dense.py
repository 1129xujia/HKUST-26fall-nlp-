from __future__ import annotations

import numpy as np

from customer_service_rag.interfaces import EmbeddingModel
from customer_service_rag.retrieval.indexing import LocalIndex
from customer_service_rag.schemas import RetrievalHit


class DenseRetriever:
    name = "dense"

    def __init__(self, index: LocalIndex, embedding: EmbeddingModel) -> None:
        self.index = index
        self.embedding = embedding

    def search(self, text: str, top_k: int, query_id: str) -> list[RetrievalHit]:
        scores = self.index.vectors @ self.embedding.encode([text])[0]
        order = np.argsort(-scores, kind="stable")
        selected = [int(i) for i in order if scores[i] > 0][:top_k]
        return [
            RetrievalHit(
                passage=self.index.passages[i], source="dense", query_id=query_id,
                score=float(scores[i]), rank=rank,
            )
            for rank, i in enumerate(selected, 1)
        ]
