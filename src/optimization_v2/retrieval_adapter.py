"""Request-local extensions around retrieval_v2; never mutate its source/config.

Raw rankings are cached before truncation. Weight/rerank changes reuse recall,
whereas query/domain/quota/corpus changes invalidate it. Model loading stays in
the original pipeline's startup code.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from retrieval_v2.retrieval.reranker import CrossEncoderReranker, NoOpReranker
from retrieval_v2.retrieval.fusion import reciprocal_rank_fusion
from retrieval_v2.retrieval.schemas import Evidence, FusedHit, RetrievalRequest

from .schemas import Capabilities, ExecutionOptions, RetrievalSnapshot


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def context_signature(evidence: list[Evidence]) -> str:
    # Order, headings and actual contents matter, not just the set of IDs.
    return fingerprint([
        (e.passage_id, e.doc_id, e.title, e.section_title, e.text, e.span_ids)
        for e in evidence
    ])


def weighted_rrf(rankings, options: ExecutionOptions) -> list[FusedHit]:
    if options.dense_weight == options.bm25_weight == 1.0:
        return reciprocal_rank_fusion(rankings, rrf_k=options.rrf_k)
    fused = {}
    for ranking in rankings:
        seen = set()
        for hit in ranking:
            key = hit.passage.passage_id
            if key in seen:
                continue
            seen.add(key)
            if hit.retriever not in {"dense", "bm25"} or hit.rank < 1:
                raise ValueError("Unsupported retriever or invalid rank")
            weight = getattr(options, f"{hit.retriever}_weight")
            if not weight:
                continue
            item = fused.setdefault(key, FusedHit(passage=copy.deepcopy(hit.passage)))
            item.rrf_score += weight / (options.rrf_k + hit.rank)
            if hit.retriever not in item.source_retrievers:
                item.source_retrievers.append(hit.retriever)
            if hit.matched_query not in item.matched_queries:
                item.matched_queries.append(hit.matched_query)
            field = f"{hit.retriever}_rank"
            previous = getattr(item, field)
            setattr(item, field, min(previous, hit.rank) if previous is not None else hit.rank)
    for item in fused.values():
        item.source_retrievers.sort()
    return sorted(fused.values(), key=lambda e: (-e.rrf_score, e.passage.passage_id))


class Part2RetrievalAdapter:
    def __init__(self, pipeline, *, corpus_version: str,
                 rerankers: dict[str, object] | None = None):
        if not corpus_version.strip():
            raise ValueError("Supply a corpus/index version for safe cache reuse")
        self.pipeline = pipeline
        self.corpus_version = corpus_version
        self.rerankers = dict(rerankers or {})
        if set(self.rerankers) - {"identity", "current_first_user_history"}:
            raise ValueError("Unknown registered reranker strategy")
        self.rerank_lock = threading.Lock()
        self.selection_lock = threading.Lock()

    @property
    def capabilities(self) -> Capabilities:
        names = [name for name in ("dense", "bm25")
                 if getattr(self.pipeline, f"{name}_retriever") is not None]
        return Capabilities(
            hybrid=bool(names), retrievers=names,
            reranking=(not isinstance(self.pipeline.reranker, NoOpReranker)
                       or bool(self.rerankers)),
        )

    def default_options(self) -> ExecutionOptions:
        cfg = self.pipeline.config
        return ExecutionOptions(
            dense_top_k=cfg.dense_top_k if cfg.use_dense else 0,
            bm25_top_k=cfg.bm25_top_k if cfg.use_bm25 else 0,
            candidate_top_k=cfg.fused_top_k, final_top_k=cfg.final_top_k,
            rrf_k=cfg.rrf_k, rerank=cfg.use_reranker and self.capabilities.reranking,
            reranker_query_strategy=cfg.reranker_query_strategy,
            reranker_use_sub_queries=cfg.reranker_use_sub_queries,
            reranker_skip_acknowledgements=cfg.reranker_skip_acknowledgements,
        )

    def request_for(self, query, options, trace_id=None) -> RetrievalRequest:
        domain = query.domain.strip().lower()
        request = RetrievalRequest(
            query=query.rewritten_query, sub_queries=list(query.sub_queries),
            domain=domain if options.domain_filter and domain != "unknown" else None,
            top_k=options.final_top_k, trace_id=trace_id,
        )
        # Reuse Part 2's boundary validation, without changing its request schema.
        return self.pipeline._validate_request(request)

    async def retrieve(self, query, options: ExecutionOptions,
                       snapshot: RetrievalSnapshot | None = None, trace_id=None):
        request = self.request_for(query, options, trace_id)
        # Copy caller-owned objects before submitting background work. A timed-out
        # sync model can finish later but cannot mutate the caller's snapshot.
        return await asyncio.to_thread(
            self._retrieve, request, options.model_copy(deep=True),
            snapshot.model_copy(deep=True) if snapshot is not None else None,
        )

    def _retrieve(self, request, options, snapshot):
        started = perf_counter()
        quotas = {name: getattr(options, f"{name}_top_k") for name in ("dense", "bm25")}
        queries = [request.query, *request.sub_queries]
        # Keep even zero-weight branch rankings so weight-only changes can reuse
        # recall. Disabling a branch completely is done with a zero quota.
        recall_key = fingerprint({
            "query": queries, "domain": request.domain, "quotas": quotas,
            "corpus_version": self.corpus_version,
            "embedding": self.pipeline.config.embedding_model,
        })
        reuse = (snapshot is not None and snapshot.recall_key == recall_key
                 and snapshot.corpus_version == self.corpus_version)
        if reuse:
            rankings = copy.deepcopy(snapshot.rankings)
        else:
            jobs = []
            for name, quota in quotas.items():
                if not quota:
                    continue
                retriever = getattr(self.pipeline, f"{name}_retriever")
                if retriever is None:
                    raise ValueError(f"Requested retrieval branch is unavailable: {name}")
                jobs.extend((retriever, text, quota) for text in queries)
            if not jobs:
                raise ValueError("No usable retrieval branches")
            with ThreadPoolExecutor(max_workers=min(
                len(jobs), self.pipeline.config.max_workers,
            )) as pool:
                futures = [pool.submit(
                    retriever.search, text, domain=request.domain, top_k=quota,
                ) for retriever, text, quota in jobs]
                rankings = [future.result() for future in futures]
        fused = weighted_rrf(rankings, options)[:options.candidate_top_k]
        candidates = copy.deepcopy(fused)
        rerank_elapsed_ms = 0.0
        rerank_model = None
        if options.rerank:
            key = options.reranker_query_strategy
            if key in self.rerankers:
                reranker = self.rerankers[key]
            elif isinstance(self.pipeline.reranker, CrossEncoderReranker):
                # A cheap wrapper over the already-loaded model. Request-local
                # settings never mutate the original CrossEncoder instance.
                original = self.pipeline.reranker
                reranker = CrossEncoderReranker(
                    original.model_name, model=original.model, batch_size=original.batch_size,
                    query_strategy=key, use_sub_queries=options.reranker_use_sub_queries,
                    skip_acknowledgements=options.reranker_skip_acknowledgements,
                )
            elif not isinstance(self.pipeline.reranker, NoOpReranker):
                raise ValueError("Register custom rerankers by strategy name")
            else:
                raise ValueError("A real reranker must be loaded or injected before reranking")
            # CrossEncoder may not be thread-safe across concurrent requests.
            with self.rerank_lock:
                rerank_started = perf_counter()
                candidates = reranker.rerank(
                    request.query,
                    request.sub_queries if options.reranker_use_sub_queries else [],
                    candidates,
                )
                rerank_elapsed_ms = (perf_counter() - rerank_started) * 1000
                rerank_model = getattr(reranker, "model_name", type(reranker).__name__)
        else:
            candidates = NoOpReranker().rerank(request.query, [], candidates)
        available = {item.passage.passage_id for item in fused}
        if ({item.passage.passage_id for item in candidates} != available
                or len(candidates) != len(available)):
            raise ValueError("Reranker must return exactly the supplied candidates")
        if any(not math.isfinite(item.rerank_score) for item in candidates):
            raise ValueError("Reranker produced a non-finite score")
        with self.selection_lock:
            selected = self.pipeline.deduplicator.select(candidates, top_k=request.top_k)
        evidence = [self.pipeline._to_evidence(item) for item in selected]
        return RetrievalSnapshot(
            recall_key=recall_key, corpus_version=self.corpus_version,
            rankings=rankings, candidates=candidates, evidence=evidence,
            reused_recall=reuse,
            branch_counts={name: sum(
                hit.retriever == name for ranking in rankings for hit in ranking
            ) for name in ("dense", "bm25")},
            elapsed_ms=(perf_counter() - started) * 1000,
            rerank_elapsed_ms=rerank_elapsed_ms, rerank_model=rerank_model,
            notes=list(self.pipeline.runtime_notes),
        )
