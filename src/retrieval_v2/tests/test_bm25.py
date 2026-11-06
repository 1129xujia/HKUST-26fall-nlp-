import pytest

from retrieval.bm25 import BM25Retriever, tokenize
from retrieval.exceptions import RetrievalModelMismatch
from tests.helpers import passage


def test_tokenize_normalizes_words() -> None:
    assert tokenize("Driver's LICENSE-Card!") == ["driver's", "license-card"]


def test_bm25_search_and_domain_filter() -> None:
    items = [passage("p1", text="renew driver license"), passage("p2", text="student grant", domain="studentaid")]
    retriever = BM25Retriever.build(items)
    assert retriever.search("driver license", top_k=1)[0].passage.passage_id == "p1"
    assert retriever.search("grant", domain="studentaid")[0].passage.passage_id == "p2"


def test_bm25_artifact_detects_corpus_change(tmp_path) -> None:
    retriever = BM25Retriever.build([passage(text="one")])
    retriever.save(tmp_path)
    with pytest.raises(RetrievalModelMismatch):
        BM25Retriever.from_artifacts(tmp_path, passages=[passage(text="two")])


def test_bm25_rejects_empty_build() -> None:
    with pytest.raises(ValueError):
        BM25Retriever.build([])


def test_bm25_does_not_return_zero_overlap_evidence() -> None:
    retriever = BM25Retriever.build([passage(text="driver license")])
    assert retriever.search("unrelatedword") == []
    assert retriever.search("!!!") == []
