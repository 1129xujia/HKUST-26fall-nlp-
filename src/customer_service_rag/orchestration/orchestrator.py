"""Build the shared runner; feature flags select single-pass or quality-loop execution."""

from __future__ import annotations

from customer_service_rag.config import AppConfig
from customer_service_rag.data.io import load_passages
from customer_service_rag.models.embeddings import build_embedding
from customer_service_rag.orchestration.loop import (
    EvaluatorFactory,
    GeneratorFactory,
    LoopOrchestrator,
    OptimizerFactory,
)
from customer_service_rag.retrieval.bm25 import BM25Retriever
from customer_service_rag.retrieval.dense import DenseRetriever
from customer_service_rag.retrieval.indexing import corpus_digest, load_index
from customer_service_rag.retrieval.pipeline import RetrievalPipeline
from customer_service_rag.reranking.reranker import CrossEncoderReranker

# Import compatibility for D2 consumers; both modes now share one implementation.
SinglePassOrchestrator = LoopOrchestrator


def build_orchestrator(
    config: AppConfig,
    *,
    evaluator_factory: EvaluatorFactory | None = None,
    optimizer_factory: OptimizerFactory | None = None,
    generator_factory: GeneratorFactory | None = None,
    mock_scenario: str | None = None,
) -> LoopOrchestrator:
    if config.pipeline == "integrated":
        if any(value is not None for value in (
            evaluator_factory, optimizer_factory, generator_factory, mock_scenario,
        )):
            raise ValueError("Scripted legacy factories require pipeline=legacy")
        from customer_service_rag.orchestration.integrated import IntegratedOrchestrator
        return IntegratedOrchestrator(config)
    embedding = build_embedding(config.models.embedding, config.project_root)
    index = load_index(config.resolve_path(config.paths.index_dir), embedding)
    expected = load_passages(config.resolve_path(config.data.passages_path))
    if corpus_digest(expected) != index.manifest["corpus_sha256"]:
        raise ValueError("Configured corpus differs from index; run cs-rag index again")
    branches = {
        "dense": DenseRetriever(index, embedding),
        "bm25": BM25Retriever(index.passages),
    }
    reranker = (
        CrossEncoderReranker(config.models.reranker, config.project_root)
        if config.retrieval.rerank else None
    )
    factories = {}
    if evaluator_factory is not None:
        factories["evaluator_factory"] = evaluator_factory
    if optimizer_factory is not None:
        factories["optimizer_factory"] = optimizer_factory
    if generator_factory is not None:
        factories["generator_factory"] = generator_factory
    return LoopOrchestrator(
        config, RetrievalPipeline(
            branches, reranker=reranker, embedding_backend=config.models.embedding.backend,
        ), mock_scenario=mock_scenario, **factories,
    )
