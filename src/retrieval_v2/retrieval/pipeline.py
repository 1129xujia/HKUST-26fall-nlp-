"""Public hybrid retrieval pipeline."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol
import warnings

from .config import RetrievalConfig
from .corpus import load_passages_jsonl
from .deduplication import EvidenceDeduplicator, NoOpDeduplicator
from .exceptions import InvalidRetrievalRequest, RetrievalIndexNotFound
from .fusion import reciprocal_rank_fusion
from .reranker import NoOpReranker, Reranker
from .schemas import Evidence, FusedHit, RetrievalRequest, SearchHit


ALLOWED_DOMAINS = {"dmv", "ssa", "va", "studentaid"}


class Retriever(Protocol):
    def search(
        self,
        query: str,
        *,
        domain: str | None = None,
        top_k: int = 20,
    ) -> list[SearchHit]: ...


class HybridRetrievalPipeline:
    """Coordinate retrieval, RRF, reranking, and deduplication."""

    def __init__(
        self,
        *,
        dense_retriever: Retriever | None,
        bm25_retriever: Retriever | None,
        config: RetrievalConfig | None = None,
        reranker: Reranker | None = None,
        deduplicator: EvidenceDeduplicator | NoOpDeduplicator | None = None,
        runtime_notes: list[str] | None = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker or NoOpReranker()
        self.deduplicator = deduplicator or (
            EvidenceDeduplicator(
                max_passages_per_doc=self.config.max_passages_per_doc,
                similarity_threshold=self.config.dedup_similarity,
            )
            if self.config.use_deduplication
            else NoOpDeduplicator()
        )
        self.runtime_notes = list(runtime_notes or [])

    @classmethod
    def from_artifacts(
        cls,
        *,
        artifact_dir: Path | str,
        config: RetrievalConfig | None = None,
        device: str | None = None,
    ) -> "HybridRetrievalPipeline":
        artifact_dir = Path(artifact_dir)
        config = config or RetrievalConfig(artifact_dir=artifact_dir)
        # 模型和索引在应用启动时加载一次；不要在每个 Graph node 调用中重复构建。
        passage_path = artifact_dir / "passages.jsonl"
        if not passage_path.exists():
            raise RetrievalIndexNotFound(f"Missing passage corpus: {passage_path}")
        passages = load_passages_jsonl(passage_path)
        runtime_notes: list[str] = []

        dense_retriever = None
        if config.use_dense:
            from .dense import DenseRetriever

            dense_retriever = DenseRetriever.from_artifacts(
                artifact_dir,
                config.embedding_model,
                passages=passages,
                query_prefix=config.dense_query_prefix,
                passage_prefix=config.dense_passage_prefix,
                device=device,
            )

        bm25_retriever = None
        if config.use_bm25:
            from .bm25 import BM25Retriever

            bm25_retriever = BM25Retriever.from_artifacts(
                artifact_dir,
                passages=passages,
            )

        reranker: Reranker = NoOpReranker()
        if config.use_reranker:
            from .reranker import CrossEncoderReranker

            try:
                reranker = CrossEncoderReranker(
                    config.reranker_model,
                    batch_size=config.reranker_batch_size,
                    device=device,
                    query_strategy=config.reranker_query_strategy,
                    use_sub_queries=config.reranker_use_sub_queries,
                    skip_acknowledgements=config.reranker_skip_acknowledgements,
                )
            except Exception as error:
                if not config.allow_reranker_fallback:
                    raise
                note = (
                    "Cross-Encoder could not be loaded; using RRF order instead: "
                    f"{type(error).__name__}: {error}"
                )
                warnings.warn(note, RuntimeWarning, stacklevel=2)
                runtime_notes.append(note)

        deduplicator = (
            EvidenceDeduplicator.from_artifacts(
                artifact_dir=artifact_dir,
                passages=passages,
                max_passages_per_doc=config.max_passages_per_doc,
                similarity_threshold=config.dedup_similarity,
            )
            if config.use_deduplication
            else NoOpDeduplicator()
        )
        return cls(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            config=config,
            reranker=reranker,
            deduplicator=deduplicator,
            runtime_notes=runtime_notes,
        )

    def retrieve(self, request: RetrievalRequest) -> list[Evidence]:
        # 主查询和子查询使用同一套召回器；最终由 RRF 融合相同 passage 的多个排名信号。
        request = self._validate_request(request)
        queries = [request.query, *request.sub_queries]
        jobs: list[tuple[Retriever, str, int]] = []

        if self.config.use_dense and self.dense_retriever is not None:
            jobs.extend(
                (self.dense_retriever, query, self.config.dense_top_k)
                for query in queries
            )
        if self.config.use_bm25 and self.bm25_retriever is not None:
            jobs.extend(
                (self.bm25_retriever, query, self.config.bm25_top_k)
                for query in queries
            )
        if not jobs:
            raise RuntimeError("No retriever is enabled or configured")

        # BM25 与 Dense 任务并行提交。Dense 模型自身有编码锁，避免底层模型并发崩溃。
        with ThreadPoolExecutor(
            max_workers=min(len(jobs), self.config.max_workers)
        ) as executor:
            futures = [
                executor.submit(
                    retriever.search,
                    query,
                    domain=request.domain,
                    top_k=top_k,
                )
                for retriever, query, top_k in jobs
            ]
            ranked_lists = [future.result() for future in futures]

        # 先融合再重排，重排器只处理较小候选集，控制延迟和内存占用。
        fused = reciprocal_rank_fusion(
            ranked_lists,
            rrf_k=self.config.rrf_k,
        )[: self.config.fused_top_k]

        reranked = self.reranker.rerank(
            request.query,
            request.sub_queries,
            fused,
        )
        selected = self.deduplicator.select(reranked, top_k=request.top_k)
        return [self._to_evidence(candidate) for candidate in selected]

    def _validate_request(self, request: RetrievalRequest) -> RetrievalRequest:
        # 在公共边界集中校验，防止错误输入深入 FAISS/BM25 后产生难定位的问题。
        if not isinstance(request.query, str):
            raise InvalidRetrievalRequest("query must be a string")
        query = request.query.strip()
        if not query:
            raise InvalidRetrievalRequest("query cannot be empty")
        domain = request.domain
        if domain is not None:
            if not isinstance(domain, str):
                raise InvalidRetrievalRequest("domain must be a string or None")
            domain = domain.strip().lower()
            if domain not in ALLOWED_DOMAINS:
                raise InvalidRetrievalRequest(f"unsupported domain: {request.domain}")
        if type(request.top_k) is not int or not 1 <= request.top_k <= 20:
            raise InvalidRetrievalRequest("top_k must be between 1 and 20")

        if not isinstance(request.sub_queries, list):
            raise InvalidRetrievalRequest("sub_queries must be a list of strings")

        sub_queries: list[str] = []
        seen = {query}
        for item in request.sub_queries:
            if not isinstance(item, str):
                raise InvalidRetrievalRequest("sub_queries must contain strings")
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                sub_queries.append(cleaned)
            if len(sub_queries) >= self.config.max_sub_queries:
                break

        return RetrievalRequest(
            query=query,
            sub_queries=sub_queries,
            domain=domain,
            top_k=request.top_k,
            trace_id=request.trace_id,
        )

    @staticmethod
    def _to_evidence(candidate: FusedHit) -> Evidence:
        passage = candidate.passage
        return Evidence(
            passage_id=passage.passage_id,
            doc_id=passage.doc_id,
            domain=passage.domain,
            title=passage.title,
            section_title=passage.section_title,
            text=passage.text,
            span_ids=passage.span_ids,
            source_retrievers=candidate.source_retrievers,
            matched_query=(candidate.matched_queries or [""])[0],
            dense_rank=candidate.dense_rank,
            bm25_rank=candidate.bm25_rank,
            rrf_score=candidate.rrf_score,
            rerank_score=candidate.rerank_score,
        )
