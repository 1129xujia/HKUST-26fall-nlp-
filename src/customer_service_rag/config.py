"""Strict YAML configuration; all relative paths are project-root relative."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import Field, PrivateAttr, model_validator

from customer_service_rag.schemas import Contract, RetrievalOptions


class DataConfig(Contract):
    passages_path: Path = Field(default_factory=lambda: Path("data/samples/passages.jsonl"))
    examples_path: Path = Field(default_factory=lambda: Path("data/samples/dialogues.jsonl"))
    source: Literal["synthetic", "multidoc2dial"] = "synthetic"
    split: Literal["mini", "train", "dev", "validation", "test"] = "mini"
    raw_dir: Path = Field(default_factory=lambda: Path("data/raw/multidoc2dial"))
    processed_dir: Path = Field(default_factory=lambda: Path("data/processed/multidoc2dial"))
    dev_ratio: float = Field(default=0.1, gt=0, lt=1)
    mini_size: int = Field(default=20, ge=1)
    chunk_size: int = Field(default=1200, ge=1)
    chunk_overlap: int = Field(default=150, ge=0)

    @model_validator(mode="after")
    def validate_chunks(self) -> DataConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class LLMConfig(Contract):
    backend: Literal["mock", "transformers", "deepseek"] = "mock"
    model: str = "mock-extractive-v1"
    revision: str | None = None
    local_path: Path | None = None
    device: Literal["cpu", "cuda"] = "cpu"
    max_new_tokens: int = Field(default=384, ge=32, le=2048)
    max_input_tokens: int = Field(default=3072, ge=256)
    num_threads: int = Field(default=4, ge=1)
    timeout_seconds: int = Field(default=45, ge=1, le=180)

    @model_validator(mode="after")
    def validate_backend(self) -> LLMConfig:
        if self.backend == "mock" and self.model != "mock-extractive-v1":
            raise ValueError("A real model name must not use the mock backend")
        if self.backend == "transformers" and (self.local_path is None or not self.revision):
            raise ValueError("Local generation requires local_path and a pinned revision")
        if self.backend == "deepseek" and self.model not in {"deepseek-flash", "deepseek-v4-pro"}:
            raise ValueError("DeepSeek backend requires a supported DeepSeek model")
        return self


class EmbeddingConfig(Contract):
    backend: Literal["hash", "transformers"] = "hash"
    model: str = "hash-token-v1"
    dimensions: int = Field(default=256, ge=8)
    revision: str | None = None
    local_path: Path | None = None
    max_length: int = Field(default=256, ge=32)
    batch_size: int = Field(default=32, ge=1)
    num_threads: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def validate_backend(self) -> EmbeddingConfig:
        if self.backend == "hash" and self.model != "hash-token-v1":
            raise ValueError("A real model name must not use hash embeddings")
        if self.backend == "transformers" and (self.local_path is None or not self.revision):
            raise ValueError("Semantic embeddings require local_path and a pinned revision")
        return self


class RerankerConfig(Contract):
    model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    revision: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    local_path: Path = Field(default_factory=lambda: Path("artifacts/models/reranker"))
    max_length: int = Field(default=512, ge=32)
    batch_size: int = Field(default=16, ge=1)
    num_threads: int = Field(default=4, ge=1)


class ModelsConfig(Contract):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)


class PlanningConfig(Contract):
    use_history: bool = True
    rewrite: bool = True
    decompose: bool = False
    max_sub_queries: int = Field(default=3, ge=1, le=5)


class QualityConfig(Contract):
    enabled: bool = False
    relevance_threshold: float = Field(default=0.7, ge=0, le=1)
    sufficiency_threshold: float = Field(default=0.7, ge=0, le=1)
    groundedness_threshold: float = Field(default=0.8, ge=0, le=1)
    dialogue_consistency_threshold: float = Field(default=0.8, ge=0, le=1)


class OptimizationConfig(Contract):
    enabled: bool = False
    max_retries: int = Field(default=2, ge=0, le=2)
    stop_if_no_new_evidence: bool = True


class BudgetConfig(Contract):
    max_latency_seconds: float = Field(default=30, gt=0)
    max_total_tokens: int = Field(default=12000, ge=1)


class EscalationConfig(Contract):
    enabled: bool = False


class PathsConfig(Contract):
    index_dir: Path = Field(default_factory=lambda: Path("artifacts/indexes/mini"))
    results_dir: Path = Field(default_factory=lambda: Path("results"))


class IntegrationConfig(Contract):
    domain_filter: bool = False
    dense_top_k: int = Field(default=50, ge=1, le=500)
    bm25_top_k: int = Field(default=50, ge=1, le=500)


class AppConfig(Contract):
    config_version: Literal["1"] = "1"
    seed: int = Field(default=42, ge=0)
    pipeline: Literal["legacy", "integrated"] = "legacy"
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    retrieval: RetrievalOptions = Field(default_factory=RetrievalOptions)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    _root: Path = PrivateAttr(default_factory=Path.cwd)

    @model_validator(mode="after")
    def validate_features(self) -> AppConfig:
        if self.optimization.enabled and not self.quality.enabled:
            raise ValueError("optimization requires quality.enabled=true")
        if self.escalation.enabled and not self.quality.enabled:
            raise ValueError("escalation requires quality.enabled=true")
        if self.planning.decompose and not self.planning.rewrite:
            raise ValueError("decomposition requires rewriting")
        if self.pipeline == "integrated" and self.retrieval.fusion != "rrf":
            raise ValueError("The integrated pipeline requires RRF fusion")
        return self

    @property
    def project_root(self) -> Path:
        return self._root

    def resolve_path(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        return path.resolve() if path.is_absolute() else (self._root / path).resolve()


def project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "configs/default.yaml"
        ).is_file():
            return candidate
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "configs/default.yaml").is_file():
        return source_root
    raise FileNotFoundError("Run inside the project or pass an explicit --config path")


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(
    config_path: str | Path | None = None,
    *,
    experiment: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    if config_path is None:
        root = project_root()
        selected = root / "configs/default.yaml"
    else:
        selected = Path(config_path).resolve()
        root = selected.parent.parent if selected.parent.name == "configs" else project_root()
    data = read_yaml(root / "configs/default.yaml")
    if selected != root / "configs/default.yaml":
        selected_data = read_yaml(selected)
        base_name = selected_data.pop("base_config", None)
        if base_name is not None:
            base_path = (selected.parent / base_name).resolve()
            if base_path == selected or base_path.parent != selected.parent:
                raise ValueError("base_config must name another file in the same config directory")
            base_data = read_yaml(base_path)
            if "base_config" in base_data:
                raise ValueError("Only one explicit base_config layer is supported")
            data = deep_merge(data, base_data)
        data = deep_merge(data, selected_data)
    if experiment is not None:
        suite = read_yaml(root / "configs/experiments.yaml")
        profiles = suite["profiles"]
        if experiment not in profiles:
            raise ValueError(f"Unknown experiment {experiment!r}; choose {', '.join(profiles)}")
        data = deep_merge(data, read_yaml(root / "configs" / suite["base"]))
        data = deep_merge(data, profiles[experiment])
    load_dotenv(root / ".env", override=False)
    environment: dict[str, Any] = {}
    if "CS_RAG_SEED" in os.environ:
        environment["seed"] = os.environ["CS_RAG_SEED"]
    if "CS_RAG_INDEX_DIR" in os.environ:
        environment["paths"] = {"index_dir": os.environ["CS_RAG_INDEX_DIR"]}
    data = deep_merge(data, environment)
    data = deep_merge(data, overrides or {})
    config = AppConfig.model_validate(data)
    config._root = root
    return config
