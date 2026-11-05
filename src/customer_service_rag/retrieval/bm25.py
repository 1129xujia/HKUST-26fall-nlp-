from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from customer_service_rag.models.embeddings import tokenize
from customer_service_rag.schemas import Passage, RetrievalHit


class BM25Retriever:
    name = "bm25"

    def __init__(self, passages: list[Passage]) -> None:
        self.passages = passages
        self.model = BM25Okapi([tokenize(f"{p.title}\n{p.text}") for p in passages])

    def search(self, text: str, top_k: int, query_id: str) -> list[RetrievalHit]:
        scores = self.model.get_scores(tokenize(text))
        order = np.argsort(-scores, kind="stable")
        selected = [int(i) for i in order if scores[i] > 0][:top_k]
        return [
            RetrievalHit(
                passage=self.passages[i], source="bm25", query_id=query_id,
                score=float(scores[i]), rank=rank,
            )
            for rank, i in enumerate(selected, 1)
        ]
