import pytest

from retrieval.config import RetrievalConfig
from retrieval.exceptions import InvalidRetrievalRequest
from retrieval.pipeline import HybridRetrievalPipeline
from retrieval.schemas import RetrievalRequest


class EmptyRetriever:
    def search(self, query, *, domain=None, top_k=20):
        return []


def pipeline() -> HybridRetrievalPipeline:
    return HybridRetrievalPipeline(
        dense_retriever=None,
        bm25_retriever=EmptyRetriever(),
        config=RetrievalConfig(use_dense=False, use_reranker=False),
    )


@pytest.mark.parametrize("query", ["", "   "])
def test_rejects_empty_query(query) -> None:
    with pytest.raises(InvalidRetrievalRequest):
        pipeline().retrieve(RetrievalRequest(query))


def test_rejects_unknown_domain() -> None:
    with pytest.raises(InvalidRetrievalRequest):
        pipeline().retrieve(RetrievalRequest("q", domain="unknown"))


def test_normalizes_domain() -> None:
    assert pipeline().retrieve(RetrievalRequest("q", domain=" DMV ")) == []


def test_rejects_top_k_outside_contract() -> None:
    with pytest.raises(InvalidRetrievalRequest):
        pipeline().retrieve(RetrievalRequest("q", top_k=21))


def test_rejects_non_string_subquery() -> None:
    with pytest.raises(InvalidRetrievalRequest):
        pipeline().retrieve(RetrievalRequest("q", sub_queries=[1]))  # type: ignore[list-item]


@pytest.mark.parametrize("value", [True, 1.5, None, "5"])
def test_rejects_non_integer_top_k(value) -> None:
    with pytest.raises(InvalidRetrievalRequest):
        pipeline().retrieve(RetrievalRequest("q", top_k=value))


def test_rejects_string_as_subquery_list() -> None:
    with pytest.raises(InvalidRetrievalRequest):
        pipeline().retrieve(RetrievalRequest("q", sub_queries="license"))
