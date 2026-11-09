import pytest

from customer_service_rag.retrieval.config import RetrievalConfig


def test_config_normalizes_paths() -> None:
    config = RetrievalConfig(data_dir="data", artifact_dir="output")
    assert config.data_dir.name == "data"
    assert config.artifact_dir.name == "output"
    assert config.use_reranker is False
    assert config.use_deduplication is False
    assert config.dense_top_k == 100
    assert config.bm25_top_k == 50
    assert config.fused_top_k == 50
    assert config.rrf_k == 30
    assert config.reranker_query_strategy == "current_first_user_history"
    assert config.reranker_use_sub_queries is False
    assert config.reranker_skip_acknowledgements is True


def test_reranker_requires_explicit_full_mode() -> None:
    assert RetrievalConfig.for_mode("rrf").use_reranker is False
    assert RetrievalConfig.for_mode("full").use_reranker is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rrf_k", 0),
        ("max_workers", 0),
        ("dedup_similarity", 1.1),
        ("max_sub_queries", -1),
        ("final_top_k", 21),
        ("reranker_query_strategy", "unknown"),
    ],
)
def test_config_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        RetrievalConfig(**{field: value})


def test_config_requires_a_retriever() -> None:
    with pytest.raises(ValueError, match="At least one retriever"):
        RetrievalConfig(use_dense=False, use_bm25=False)
