from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from uuid import uuid4
from importlib.resources import files
from pathlib import Path

import yaml

from customer_service_rag import __version__
from customer_service_rag.config import AppConfig
from customer_service_rag.schemas import RunResult


def write_run(
    config: AppConfig, result: RunResult, *, scenario_definition: dict | None = None,
) -> Path:
    directory = config.resolve_path(config.paths.results_dir) / result.run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8",
    )
    try:
        code_version = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.project_root,
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=config.project_root,
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        code_version, dirty = "unknown", None
    metadata = {
        "package_version": __version__, "python": platform.python_version(),
        "platform": platform.platform(), "code_version": code_version,
        "worktree_dirty": dirty, "is_mock": result.is_mock,
        "source": config.data.source, "split": config.data.split,
        "note": "Development run; dataset split and model backends determine its scope.",
        "model_backend": result.model_backend,
        "mock_scenario": result.mock_scenario,
        "quality_mode": (
            "scripted" if result.mock_scenario else (
                "mock_lexical" if result.model_backend == "mock" else "local_llm"
            )
        ) if config.quality.enabled else "disabled",
        "models": config.models.model_dump(mode="json"),
        "data_sha256": {
            key: hashlib.sha256(config.resolve_path(path).read_bytes()).hexdigest()
            for key, path in {
                "passages": config.data.passages_path,
                "examples": config.data.examples_path,
            }.items()
        },
        "index_manifest": json.loads(
            (config.resolve_path(config.paths.index_dir) / "manifest.json").read_text(
                encoding="utf-8"
            )
        ),
        "prompt_sha256": {
            name: hashlib.sha256(
                files("customer_service_rag").joinpath(f"prompts/{name}.jinja2").read_bytes()
            ).hexdigest()
            for name in ["planning", "generation", "evaluation", "optimization"]
        },
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (directory / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    (directory / "predictions.jsonl").write_text(
        result.model_dump_json() + "\n", encoding="utf-8",
    )
    (directory / "traces.jsonl").write_text(
        "".join(event.model_dump_json() + "\n" for event in result.trace), encoding="utf-8",
    )
    for filename, records in {
        "retrieval.jsonl": result.retrieval_history,
        "feedback.jsonl": result.feedback,
        "actions.jsonl": result.actions,
        "candidates.jsonl": result.candidate_history,
    }.items():
        (directory / filename).write_text(
            "".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8",
        )
    (directory / "model_calls.jsonl").write_text(
        "".join(json.dumps(call, ensure_ascii=False) + "\n" for call in result.model_calls),
        encoding="utf-8",
    )
    if scenario_definition is not None:
        (directory / "scenario.json").write_text(
            json.dumps(scenario_definition, indent=2, ensure_ascii=False), encoding="utf-8",
        )
    return directory


def write_batch(config: AppConfig, results: list[RunResult]) -> Path:
    directory = config.resolve_path(config.paths.results_dir) / ("batch_" + uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8",
    )
    (directory / "predictions.jsonl").write_text(
        "".join(result.model_dump_json() + "\n" for result in results), encoding="utf-8",
    )
    (directory / "batch.json").write_text(json.dumps({
        "run_ids": [result.run_id for result in results],
        "source": config.data.source, "split": config.data.split,
        "is_mock": any(result.is_mock for result in results),
        "reference_file": str(config.resolve_path(config.data.examples_path)),
    }, indent=2), encoding="utf-8")
    return directory
