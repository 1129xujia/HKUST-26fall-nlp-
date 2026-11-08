"""Shared configuration for selectable segmentation and model chains."""

from pathlib import Path

from .config import RetrievalConfig


def profile_config(*, segmentation="section", chain="rerank",
                   index_root="artifacts/indexes", artifact_dir=None,
                   embedding_model=None, reranker_model=None):
    if chain not in {"rerank", "finetuned"}:
        raise ValueError("chain must be rerank or finetuned")
    if embedding_model is None:
        local = Path("local_models/bge-base-en-v1.5")
        embedding_model = (
            "models/bge-base-en-v1.5-struct-ft/checkpoint-2000" if chain == "finetuned"
            else str(local) if local.is_dir() else RetrievalConfig.embedding_model
        )
    if chain == "finetuned" and not Path(embedding_model).is_dir():
        raise ValueError(f"Fine-tuned model directory is missing: {embedding_model}; retrieve the checkpoint with Git LFS first")
    if reranker_model is None:
        local = Path("local_models/ms-marco-MiniLM-L6-v2")
        reranker_model = str(local) if local.is_dir() else RetrievalConfig.reranker_model
    return RetrievalConfig.for_mode(
        chain, artifact_dir=artifact_dir or Path(index_root) / segmentation / chain,
        segmentation=segmentation, embedding_model=embedding_model,
        reranker_model=reranker_model, allow_reranker_fallback=False,
    )
