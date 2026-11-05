from __future__ import annotations

import asyncio
from pathlib import Path

from customer_service_rag.config import RerankerConfig
from customer_service_rag.models.runtime import load_runtime
from customer_service_rag.schemas import QueryPlan, RetrievalCandidate


class CrossEncoderReranker:
    def __init__(self, config: RerankerConfig, root: Path) -> None:
        self.config = config
        path = config.local_path
        path = path if path.is_absolute() else root / path
        self.model_name = f"{config.model}@{config.revision}"
        self.runtime = load_runtime(
            str(path.resolve()), "reranker", config.model, config.revision, config.num_threads,
        )

    def _rank(self, query: QueryPlan, candidates: list[RetrievalCandidate]):
        runtime = self.runtime
        scores = []
        with runtime.lock, runtime.torch.inference_mode():
            for start in range(0, len(candidates), self.config.batch_size):
                batch = candidates[start:start + self.config.batch_size]
                inputs = runtime.tokenizer(
                    [query.rewritten_query] * len(batch),
                    [f"{item.passage.title}\n{item.passage.text}" for item in batch],
                    padding=True, truncation="only_second", max_length=self.config.max_length,
                    return_tensors="pt",
                )
                logits = runtime.model(**inputs).logits
                if logits.shape[-1] != 1:
                    raise ValueError("This reranker expects a single relevance logit")
                scores.extend(logits[:, 0].cpu().tolist())
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)
        return sorted(candidates, key=lambda item: (-item.rerank_score, item.passage.citation_id))

    async def rerank(
        self, query: QueryPlan, candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        return await asyncio.to_thread(self._rank, query, candidates)
