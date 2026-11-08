from types import SimpleNamespace

import pytest

from retrieval.scripts.evaluate_profile import ranking_metrics, evaluate


def test_profile_metrics_multi_positive_and_cutoff():
    result = ranking_metrics(["wrong", "a"] + [f"n{i}" for i in range(8)] + ["b"], ["a", "b"])
    assert result["hit@10"] == 1
    assert result["recall@10"] == .5
    assert result["mrr@10"] == .5
    assert result["ndcg@10"] == pytest.approx((1 / 1.584962500721156) / (1 + 1 / 1.584962500721156))


def test_profile_metrics_reject_bad_gold_and_duplicate_rankings():
    with pytest.raises(ValueError):
        ranking_metrics(["a"], [])
    with pytest.raises(ValueError):
        ranking_metrics(["a", "a"], ["a"])


def test_profile_evaluation_rejects_small_corpus_before_retrieving():
    pipeline = SimpleNamespace(dense_retriever=SimpleNamespace(passages=[]))
    with pytest.raises(ValueError, match="Gold is missing"):
        evaluate(pipeline, [{"sample_id": "q", "query": "question", "gold": ["missing"]}])
