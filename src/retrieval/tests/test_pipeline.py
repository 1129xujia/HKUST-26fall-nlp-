from dataclasses import replace

from retrieval.config import RetrievalConfig
from retrieval.deduplication import NoOpDeduplicator
from retrieval.pipeline import HybridRetrievalPipeline
from retrieval.schemas import RetrievalRequest
from .helpers import hit


class FakeRetriever:
    def __init__(self, name):
        self.name = name
        self.queries = []

    def search(self, query, *, domain=None, top_k=20):
        self.queries.append(query)
        return [replace(hit("p", self.name, 1), matched_query=query)]


def test_pipeline_returns_evidence() -> None:
    config = replace(RetrievalConfig(), use_reranker=False)
    pipeline = HybridRetrievalPipeline(
        dense_retriever=FakeRetriever("dense"), bm25_retriever=FakeRetriever("bm25"), config=config
    )
    result = pipeline.retrieve(RetrievalRequest("query", top_k=1))
    assert result[0].passage_id == "p"
    assert result[0].source_retrievers == ["bm25", "dense"]


def test_pipeline_defaults_to_no_extra_deduplication() -> None:
    pipeline = HybridRetrievalPipeline(
        dense_retriever=FakeRetriever("dense"),
        bm25_retriever=FakeRetriever("bm25"),
    )
    assert isinstance(pipeline.deduplicator, NoOpDeduplicator)


def test_pipeline_runs_subqueries() -> None:
    retriever = FakeRetriever("bm25")
    pipeline = HybridRetrievalPipeline(
        dense_retriever=None,
        bm25_retriever=retriever,
        config=RetrievalConfig(use_dense=False, use_reranker=False),
    )
    result = pipeline.retrieve(RetrievalRequest("main", ["sub"], top_k=1))
    assert set(retriever.queries) == {"main", "sub"}
    assert result[0].matched_query in {"main", "sub"}
