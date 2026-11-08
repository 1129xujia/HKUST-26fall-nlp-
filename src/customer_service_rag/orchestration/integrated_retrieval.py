"""Run Part 2 algorithms on the shared, checksum-validated project index.

The corpus keeps its original citation IDs, so existing grounding/evaluation
files remain usable. Part 2's standalone FAISS workflow is still independent.
"""

from __future__ import annotations

from threading import Lock

import numpy as np

from customer_service_rag.data.io import load_passages
from customer_service_rag.models.embeddings import build_embedding
from customer_service_rag.models.runtime import load_runtime
from customer_service_rag.retrieval.indexing import corpus_digest, load_index
from customer_service_rag.schemas import (
    RetrievalCandidate, RetrievalHit, RetrievalOptions, RetrievalResult,
)
from optimization_v2.retrieval_adapter import Part2RetrievalAdapter
from optimization_v2.schemas import ExecutionOptions
from retrieval.bm25 import BM25Retriever
from retrieval.config import RetrievalConfig
from retrieval.pipeline import HybridRetrievalPipeline
from retrieval.reranker import CrossEncoderReranker
from retrieval.schemas import Passage, SearchHit


class SharedDenseRetriever:
    def __init__(self, passages, index, embedding):
        self.passages, self.index, self.embedding = passages, index, embedding
        self.lock = Lock()

    def search(self, query, *, domain=None, top_k=20):
        with self.lock:
            vector = self.embedding.encode([query])[0]
        scores = self.index.vectors @ vector
        rows = [int(row) for row in np.argsort(-scores, kind="stable")
                if (domain is None or self.passages[int(row)].domain == domain)
                and (getattr(self.embedding, "backend", "hash") != "hash" or scores[row] > 0)]
        return [SearchHit(self.passages[row], "dense", rank, float(scores[row]), query)
                for rank, row in enumerate(rows[:top_k], 1)]


class LocalPairPredictor:
    """Use verified local weights through Part 2's actual CrossEncoder reranker."""

    def __init__(self, config, root):
        self.config = config
        self.runtime = load_runtime(
            str((root / config.local_path).resolve()), "reranker",
            config.model, config.revision, config.num_threads,
        )

    def predict(self, pairs, *, batch_size=16, show_progress_bar=False):
        runtime, scores = self.runtime, []
        with runtime.lock, runtime.torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start:start + batch_size]
                inputs = runtime.tokenizer(
                    [pair[0] for pair in batch], [pair[1] for pair in batch],
                    padding=True, truncation="only_second", max_length=self.config.max_length,
                    return_tensors="pt",
                )
                logits = runtime.model(**inputs).logits
                if logits.shape[-1] != 1:
                    raise ValueError("The configured reranker must return one relevance logit")
                scores.extend(logits[:, 0].cpu().tolist())
        return np.asarray(scores)


def legacy_options(options: ExecutionOptions) -> RetrievalOptions:
    branches = [name for name in ("dense", "bm25")
                if getattr(options, f"{name}_top_k") > 0
                and getattr(options, f"{name}_weight") > 0]
    return RetrievalOptions(
        branches=branches,
        per_branch_top_k=max(options.dense_top_k, options.bm25_top_k), fusion="rrf",
        rrf_k=options.rrf_k, candidate_top_k=options.candidate_top_k,
        final_top_k=options.final_top_k, rerank=options.rerank,
    )


class IntegratedRetriever:
    def __init__(self, config):
        self.config = config
        self.embedding = build_embedding(config.models.embedding, config.project_root)
        index = load_index(config.resolve_path(config.paths.index_dir), self.embedding)
        expected = load_passages(config.resolve_path(config.data.passages_path))
        if corpus_digest(expected) != index.manifest["corpus_sha256"]:
            raise ValueError("Configured corpus differs from index; run cs-rag index again")
        self.passages = {p.citation_id: p for p in index.passages}
        converted = [Passage(
            passage_id=p.citation_id, doc_id=p.doc_id, domain=p.domain, title=p.title,
            section_title="", parent_titles=[], text=p.text, span_ids=p.span_ids,
            start=p.char_start, end=p.char_end,
        ) for p in index.passages]
        branches = config.retrieval.branches
        dense = SharedDenseRetriever(converted, index, self.embedding) if "dense" in branches else None
        bm25 = BM25Retriever.build(converted) if "bm25" in branches else None
        reranker = None
        if config.retrieval.rerank:
            settings = config.models.reranker
            reranker = CrossEncoderReranker(
                f"{settings.model}@{settings.revision}",
                model=LocalPairPredictor(settings, config.project_root),
                batch_size=settings.batch_size,
            )
        pipeline = HybridRetrievalPipeline(
            dense_retriever=dense, bm25_retriever=bm25, reranker=reranker,
            config=RetrievalConfig(
                embedding_model=self.embedding.model_name,
                use_dense=dense is not None, use_bm25=bm25 is not None,
                use_reranker=reranker is not None, use_deduplication=False,
                dense_top_k=config.integration.dense_top_k,
                bm25_top_k=config.integration.bm25_top_k,
                fused_top_k=config.retrieval.candidate_top_k,
                final_top_k=config.retrieval.final_top_k,
                rrf_k=config.retrieval.rrf_k,
            ),
        )
        self.adapter = Part2RetrievalAdapter(pipeline, corpus_version=index.manifest["corpus_sha256"])
        self.reranker = reranker

    def options(self, options: RetrievalOptions) -> ExecutionOptions:
        return ExecutionOptions(
            dense_top_k=self.config.integration.dense_top_k if "dense" in options.branches else 0,
            bm25_top_k=self.config.integration.bm25_top_k if "bm25" in options.branches else 0,
            candidate_top_k=options.candidate_top_k, final_top_k=options.final_top_k,
            rrf_k=options.rrf_k, rerank=options.rerank,
            # Standalone rewritten queries already contain their visible context.
            reranker_query_strategy="identity", reranker_use_sub_queries=False,
            domain_filter=self.config.integration.domain_filter,
        )

    def result(self, snapshot) -> RetrievalResult:
        by_passage = {}
        hits = []
        for ranking in snapshot.rankings:
            for hit in ranking:
                converted = RetrievalHit(
                    passage=self.passages[hit.passage.passage_id], source=hit.retriever,
                    query_id=hit.matched_query, score=hit.raw_score, rank=hit.rank,
                )
                hits.append(converted)
                by_passage.setdefault(hit.passage.passage_id, []).append(converted)
        return RetrievalResult(
            passages=[self.passages[e.passage_id] for e in snapshot.evidence],
            candidates=[RetrievalCandidate(
                passage=self.passages[c.passage.passage_id],
                hits=by_passage.get(c.passage.passage_id, []), fusion_score=c.rrf_score,
                rerank_score=c.rerank_score,
            ) for c in snapshot.candidates], hits=hits,
            elapsed_ms=snapshot.elapsed_ms,
            embedding_backend=self.config.models.embedding.backend,
            rerank_elapsed_ms=snapshot.rerank_elapsed_ms, rerank_model=snapshot.rerank_model,
        )

    async def retrieve(self, query, options):
        return self.result(await self.adapter.retrieve(query, self.options(options)))
