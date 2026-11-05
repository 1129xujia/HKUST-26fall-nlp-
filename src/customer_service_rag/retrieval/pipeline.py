from __future__ import annotations

from time import perf_counter

from customer_service_rag.fusion.deduplication import interleave
from customer_service_rag.fusion.rrf import reciprocal_rank_fusion
from customer_service_rag.interfaces import SearchBranch
from customer_service_rag.reranking.reranker import CrossEncoderReranker
from customer_service_rag.retrieval.parallel import parallel_search
from customer_service_rag.schemas import QueryPlan, RetrievalOptions, RetrievalResult


class RetrievalPipeline:
    def __init__(
        self, branches: dict[str, SearchBranch], *,
        reranker: CrossEncoderReranker | None = None, embedding_backend: str = "hash",
    ) -> None:
        self.branches = branches
        self.reranker = reranker
        self.embedding_backend = embedding_backend

    async def retrieve(self, query: QueryPlan, options: RetrievalOptions) -> RetrievalResult:
        if options.rerank and self.reranker is None:
            raise ValueError("Reranking is enabled but no reranker was configured")
        start = perf_counter()
        rankings, timings = await parallel_search(
            [self.branches[name] for name in options.branches],
            query.retrieval_queries(),
            options.per_branch_top_k,
        )
        candidates = (
            reciprocal_rank_fusion(rankings, options.rrf_k)
            if options.fusion == "rrf" else interleave(rankings)
        )[:options.candidate_top_k]
        rerank_elapsed_ms = 0.0
        if options.rerank and candidates:
            rerank_start = perf_counter()
            candidates = await self.reranker.rerank(query, candidates)
            rerank_elapsed_ms = (perf_counter() - rerank_start) * 1000
        return RetrievalResult(
            passages=[item.passage for item in candidates[:options.final_top_k]],
            candidates=candidates,
            hits=[hit for ranking in rankings for hit in ranking],
            elapsed_ms=(perf_counter() - start) * 1000,
            branch_elapsed_ms=timings,
            embedding_backend=self.embedding_backend,
            rerank_elapsed_ms=rerank_elapsed_ms,
            rerank_model=self.reranker.model_name if options.rerank else None,
        )
