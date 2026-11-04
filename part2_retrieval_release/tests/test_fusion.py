import pytest

from retrieval.fusion import reciprocal_rank_fusion
from tests.helpers import hit


def test_rrf_rewards_agreement() -> None:
    fused = reciprocal_rank_fusion([[hit("shared", "dense", 1)], [hit("shared", "bm25", 1), hit("other", "bm25", 2)]])
    assert fused[0].passage.passage_id == "shared"
    assert fused[0].source_retrievers == ["bm25", "dense"]


def test_rrf_duplicate_in_one_list_does_not_double_count() -> None:
    once = reciprocal_rank_fusion([[hit("p", "dense", 1)]])[0].rrf_score
    twice = reciprocal_rank_fusion([[hit("p", "dense", 1), hit("p", "dense", 2)]])[0].rrf_score
    assert twice == once


def test_rrf_rejects_invalid_k() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], rrf_k=0)
