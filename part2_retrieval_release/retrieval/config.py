"""Configuration for the retrieval subsystem."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetrievalConfig:
    # 线上默认值来自 validation 消融。实验新参数应先通过评估，再修改这里的默认值。
    data_dir: Path = Path("multidoc2dial")
    artifact_dir: Path = Path("artifacts")
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    # Cross-Encoder performs best when the latest user question is placed first
    # and previous user questions provide context without long agent answers.
    reranker_query_strategy: str = "current_first_user_history"
    # Sub-queries still expand first-stage recall, but are not allowed to
    # dominate reranking until a dedicated sub-query scoring ablation exists.
    reranker_use_sub_queries: bool = False
    # Short acknowledgements lose their meaning when agent questions are
    # removed. Preserve RRF order for a narrow, explicit acknowledgement list.
    reranker_skip_acknowledgements: bool = True
    dense_query_prefix: str = "Represent this sentence for searching relevant passages: "
    dense_passage_prefix: str = ""
    use_dense: bool = True
    use_bm25: bool = True
    # Keep the production path RRF-only while first-stage recall is being tuned.
    # The Cross-Encoder remains available through for_mode("full").
    use_reranker: bool = False
    # RRF already merges identical passage IDs. Additional semantic/per-document
    # filtering is disabled because validation showed that it removed Gold evidence.
    use_deduplication: bool = False
    allow_reranker_fallback: bool = True
    dense_top_k: int = 100
    bm25_top_k: int = 50
    fused_top_k: int = 50
    final_top_k: int = 6
    rrf_k: int = 30
    reranker_batch_size: int = 16
    max_sub_queries: int = 4
    max_passages_per_doc: int = 2
    dedup_similarity: float = 0.93
    max_workers: int = 8

    @classmethod
    def for_mode(
        cls,
        mode: str,
        *,
        artifact_dir: Path | str = Path("artifacts"),
        allow_reranker_fallback: bool = True,
    ) -> "RetrievalConfig":
        # mode 是给调用方的稳定开关，避免上游了解各 retriever 的内部组合方式。
        if mode not in {"bm25", "dense", "rrf", "full"}:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
        return cls(
            artifact_dir=Path(artifact_dir),
            use_dense=mode != "bm25",
            use_bm25=mode != "dense",
            use_reranker=mode == "full",
            allow_reranker_fallback=allow_reranker_fallback,
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "artifact_dir", Path(self.artifact_dir))
        for name in (
            "dense_top_k",
            "bm25_top_k",
            "fused_top_k",
            "rrf_k",
            "reranker_batch_size",
            "max_sub_queries",
            "max_passages_per_doc",
            "max_workers",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 1 <= self.final_top_k <= 20:
            raise ValueError("final_top_k must be between 1 and 20")
        if not 0.0 <= self.dedup_similarity <= 1.0:
            raise ValueError("dedup_similarity must be between 0 and 1")
        if not (self.use_dense or self.use_bm25):
            raise ValueError("At least one retriever must be enabled")
        if self.reranker_query_strategy not in {
            "identity",
            "current_first_user_history",
        }:
            raise ValueError("Unsupported reranker_query_strategy")
