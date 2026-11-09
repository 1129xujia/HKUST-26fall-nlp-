"""Compact, reproducible run records; model calls remain inside result.json."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib.resources import files
from pathlib import Path
from uuid import uuid4

import yaml

from customer_service_rag import __version__
from customer_service_rag.config import AppConfig
from customer_service_rag.schemas import RunResult


def _code_identity(root: Path):
    command = ["git", "-c", f"safe.directory={root.as_posix()}"]
    try:
        revision = subprocess.run(command + ["rev-parse", "HEAD"], cwd=root,
            capture_output=True, text=True, check=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(command + ["status", "--porcelain"], cwd=root,
            capture_output=True, text=True, check=True, timeout=5).stdout.strip())
        return revision, dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", None


def _component_hashes():
    components = {
        "customer_service_rag": ["planning/planner.py", "generation/generator.py",
            "orchestration/orchestrator.py", "orchestration/retrieval.py", "models/deepseek_llm.py"],
        "customer_service_rag.optimization": ["adapters.py", "controller.py", "router.py", "budget.py", "retrieval_adapter.py"],
        "customer_service_rag.retrieval": ["pipeline.py", "dense.py", "bm25.py", "fusion.py", "reranker.py", "corpus.py"],
    }
    return {f"{package}/{name}": hashlib.sha256(files(package).joinpath(name).read_bytes()).hexdigest()
            for package, names in components.items() for name in names}


def _save_config(config: AppConfig, directory: Path):
    (directory / "config.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8")


def write_run(config: AppConfig, result: RunResult) -> Path:
    directory = config.resolve_path(config.paths.results_dir) / result.run_id
    directory.mkdir(parents=True, exist_ok=False)
    _save_config(config, directory)
    revision, dirty = _code_identity(config.project_root)
    metadata = {
        "package_version": __version__, "python": platform.python_version(),
        "platform": platform.platform(), "code_version": revision, "worktree_dirty": dirty,
        "is_mock": result.is_mock, "source": config.data.source, "split": config.data.split,
        "model_backend": result.model_backend, "pipeline_backend": result.pipeline_backend,
        "quality_mode": result.integration.get("quality_mode", "disabled"),
        "models": config.models.model_dump(mode="json"),
        "retrieval": config.retrieval.model_dump(mode="json"),
        "index_manifest": result.integration.get("corpus", {}),
        "data_sha256": {key: hashlib.sha256(config.resolve_path(path).read_bytes()).hexdigest()
            for key, path in {"passages": config.data.passages_path,
                              "examples": config.data.examples_path}.items()
            if config.resolve_path(path).is_file()},
        "prompt_sha256": {name: hashlib.sha256(files("customer_service_rag").joinpath(
            f"prompts/{name}.jinja2").read_bytes()).hexdigest() for name in ("planning", "evaluation")},
        "component_sha256": _component_hashes(),
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (directory / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return directory


def write_batch(config: AppConfig, results: list[RunResult]) -> Path:
    directory = config.resolve_path(config.paths.results_dir) / ("batch_" + uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    _save_config(config, directory)
    (directory / "predictions.jsonl").write_text(
        "".join(result.model_dump_json() + "\n" for result in results), encoding="utf-8")
    (directory / "batch.json").write_text(json.dumps({
        "run_ids": [r.run_id for r in results], "source": config.data.source,
        "split": config.data.split, "is_mock": any(r.is_mock for r in results),
        "reference_file": str(config.resolve_path(config.data.examples_path)),
    }, indent=2), encoding="utf-8")
    return directory
