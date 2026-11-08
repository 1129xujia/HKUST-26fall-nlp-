import numpy as np
import pytest

from retrieval.reranker import CrossEncoderReranker, NoOpReranker
from retrieval.schemas import FusedHit
from .helpers import passage


class FakeModel:
    def predict(self, pairs, **kwargs):
        return np.arange(len(pairs), dtype=float)


def test_cross_encoder_uses_best_query() -> None:
    item = FusedHit(passage("p"), matched_queries=["main"], rrf_score=0.1)
    result = CrossEncoderReranker("fake", model=FakeModel()).rerank("main", ["sub"], [item])
    assert result[0].matched_queries[0] == "sub"
    assert result[0].rerank_score == 1.0


def test_cross_encoder_can_rerank_with_current_user_first() -> None:
    class RecordingModel:
        def __init__(self):
            self.pairs = None

        def predict(self, pairs, **kwargs):
            self.pairs = pairs
            return [0.5] * len(pairs)

    model = RecordingModel()
    reranker = CrossEncoderReranker(
        "fake",
        model=model,
        query_strategy="current_first_user_history",
        use_sub_queries=False,
    )
    item = FusedHit(passage("p"), matched_queries=["original"])
    result = reranker.rerank(
        "User: old question\nAgent: long answer\nUser: current question",
        ["unused sub-query"],
        [item],
    )
    assert model.pairs == [("current question\nold question", item.passage.index_text)]
    assert result[0].matched_queries[0] == "current question\nold question"


def test_cross_encoder_can_preserve_rrf_for_acknowledgement() -> None:
    class FailIfCalledModel:
        def predict(self, pairs, **kwargs):
            raise AssertionError("model should not score low-information acknowledgement")

    item = FusedHit(passage("p"), rrf_score=0.25)
    reranker = CrossEncoderReranker(
        "fake",
        model=FailIfCalledModel(),
        query_strategy="current_first_user_history",
        skip_acknowledgements=True,
    )
    result = reranker.rerank(
        "User: earlier question\nAgent: clarification\nUser: Yes.", [], [item]
    )
    assert result == [item]
    assert result[0].rerank_score == 0.25


def test_noop_uses_rrf_score() -> None:
    item = FusedHit(passage("p"), rrf_score=0.5)
    assert NoOpReranker().rerank("q", [], [item])[0].rerank_score == 0.5


def test_cross_encoder_handles_empty_candidates() -> None:
    assert CrossEncoderReranker("fake", model=FakeModel()).rerank("q", [], []) == []


def test_cross_encoder_rejects_non_finite_scores() -> None:
    class InvalidModel:
        def predict(self, pairs, **kwargs):
            return [float("nan")] * len(pairs)

    reranker = CrossEncoderReranker("fake", model=InvalidModel())
    with pytest.raises(RuntimeError, match="non-finite"):
        reranker.rerank("q", [], [FusedHit(passage())])
