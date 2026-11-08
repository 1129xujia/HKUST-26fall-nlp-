"""Configuration for the retrieval subsystem."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetrievalConfig:
    data_dir: Path = Path("multidoc2dial")
    artifact_dir: Path = Path("artifacts")
    # auto 从索引 metadata 选择后端；旧 BGE 索引不需要重新生成。
    dense_backend: str = "auto"
    dpr_model_dir: Path | None = None
    segmentation: str | None = None
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
        segmentation: str | None = None,
    ) -> "RetrievalConfig":
        if mode not in {"bm25", "dense", "rrf", "full"}:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
        return cls(
            artifact_dir=Path(artifact_dir),
            use_dense=mode != "bm25",
            use_bm25=mode != "dense",
            use_reranker=mode == "full",
            allow_reranker_fallback=allow_reranker_fallback,
            segmentation=segmentation,
        )

    @classmethod
    def for_dpr_baseline(
        cls, *, artifact_dir: Path | str = Path("artifacts_dpr"), mode: str = "dense",
        model_dir: Path | str | None = None,
        segmentation: str | None = None,
    ) -> "RetrievalConfig":
        """正式基线只开 DPR；显式传 rrf/full 才开启其他检索/重排组件。"""
        from dataclasses import replace

        return replace(
            cls.for_mode(mode, artifact_dir=artifact_dir, allow_reranker_fallback=False, segmentation=segmentation),
            dense_backend="dpr", dpr_model_dir=Path(model_dir) if model_dir else None,
            dense_query_prefix="", dense_passage_prefix="",
            dense_top_k=200, fused_top_k=200,
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "artifact_dir", Path(self.artifact_dir))
        if self.dpr_model_dir is not None:
            object.__setattr__(self, "dpr_model_dir", Path(self.dpr_model_dir))
        if self.dense_backend not in {"auto", "sentence_transformer", "dpr"}:
            raise ValueError("Unsupported dense_backend")
        if self.segmentation not in {None, "section", "struct"}:
            raise ValueError("segmentation must be 'section', 'struct' or None")
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
