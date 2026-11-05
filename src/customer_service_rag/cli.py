"""Executable D1-D3 offline development commands. No network calls are made."""

from __future__ import annotations

import asyncio
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Optional

import typer
import yaml
from pydantic.json_schema import models_json_schema

from customer_service_rag import __version__, schemas
from customer_service_rag.config import AppConfig, load_config
from customer_service_rag.data.io import load_examples, load_passages
from customer_service_rag.data.multidoc2dial import inspect_dataset
from customer_service_rag.data.preprocessing import prepare_mini
from customer_service_rag.models.embeddings import build_embedding
from customer_service_rag.observability.tracing import write_batch, write_run
from customer_service_rag.orchestration.orchestrator import build_orchestrator
from customer_service_rag.orchestration.scenarios import (
    MockScenario,
    load_scenarios,
    scenario_factories,
)
from customer_service_rag.schemas import EvaluationExample, QueryPlan, ResponseStatus, RunResult

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Customer-service RAG: local models, hybrid retrieval and bounded quality loops.",
)


def emit(value: object) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2))


@app.command("config")
def show_config(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    experiment: Optional[str] = typer.Option(None, "--experiment"),
) -> None:
    """Print validated, fully expanded configuration."""
    settings = load_config(config, experiment=experiment)
    typer.echo(yaml.safe_dump(settings.model_dump(mode="json"), sort_keys=False))


@app.command()
def doctor(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Show local runtime, offline backends, data counts and index availability."""
    settings = load_config(config)
    passages = load_passages(settings.resolve_path(settings.data.passages_path))
    examples = load_examples(settings.resolve_path(settings.data.examples_path))
    emit({
        "package_version": __version__,
        "python": platform.python_version(),
        "executable": sys.executable,
        "project_root": str(settings.project_root),
        "llm_backend": settings.models.llm.backend,
        "llm_model": settings.models.llm.model,
        "embedding_backend": settings.models.embedding.backend,
        "embedding_model": settings.models.embedding.model,
        "gpu_required": settings.models.llm.device == "cuda",
        "external_api_required": False,
        "quality_loop_enabled": settings.quality.enabled,
        "optimization_enabled": settings.optimization.enabled,
        "escalation_enabled": settings.escalation.enabled,
        "data_source": settings.data.source,
        "passages": len(passages),
        "examples": len(examples),
        "index_exists": (settings.resolve_path(settings.paths.index_dir) / "manifest.json").is_file(),
        "dependencies": {
            name: version(name) for name in ["pydantic", "PyYAML", "typer", "numpy", "rank-bm25"]
        },
    })


@app.command("schema")
def export_schema(output: Optional[Path] = typer.Option(None, "--output", "-o")) -> None:
    """Export version 1 JSON Schema for the shared contracts and configuration."""
    contracts = [
        schemas.DialogueInput, schemas.QueryPlan, schemas.Passage, schemas.RetrievalHit,
        schemas.RetrievalOptions, schemas.RetrievalResult, schemas.GeneratedResponse,
        schemas.EvaluationFeedback, schemas.OptimizationAction, schemas.RunState,
        schemas.RunResult, schemas.EvaluationExample, AppConfig, MockScenario,
    ]
    _, document = models_json_schema(
        [(contract, "validation") for contract in contracts],
        title="Customer Service RAG Contracts v1",
    )
    document["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    if output is None:
        emit(document)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
        typer.echo(f"Schema saved: {output.resolve()}")


@app.command()
def samples(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """List authored development scenarios without exposing gold labels to agents."""
    settings = load_config(config)
    examples = load_examples(settings.resolve_path(settings.data.examples_path))
    emit([{
        "sample_id": example.sample_id,
        "message": example.dialogue.current_message,
        "history_turns": len(example.dialogue.dialogue_history),
        "expected_behavior": example.expected_behavior,
    } for example in examples])


@app.command()
def preprocess(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Prepare synthetic examples or the official dataset with grounding alignment."""
    settings = load_config(config)
    if settings.data.source == "multidoc2dial":
        from customer_service_rag.data.official import prepare_official
        emit(prepare_official(settings))
    else:
        emit(prepare_mini(settings))


@app.command("download-data")
def download_data(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Download the official MultiDoc2Dial archive and record source checksums."""
    from customer_service_rag.assets import download_dataset
    settings = load_config(config)
    manifest = download_dataset(settings.project_root)
    emit({key: value for key, value in manifest.items() if key != "files"})


@app.command("download-models")
def download_model_assets(
    endpoint: str = typer.Option("https://hf-mirror.com", "--endpoint"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Download pinned, safetensors-only local model assets; no inference service needed."""
    from customer_service_rag.assets import download_models
    registry = download_models(load_config(config).project_root, endpoint)
    emit({key: {
        "repo_id": entry["repo_id"], "revision": entry["revision"],
        "path": entry["path"], "license": entry["license"],
    } for key, entry in registry.items()})


@app.command("inspect-data")
def inspect_data(
    documents: Path = typer.Option(..., "--documents", exists=True, dir_okay=False),
    dialogues: Path = typer.Option(..., "--dialogues", exists=True, dir_okay=False),
) -> None:
    """Inspect local official doc_data/dial_data JSON containers (no download)."""
    emit(inspect_dataset(documents, dialogues))


@app.command()
def index(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Build a persistent temporary hash-vector index and saved BM25 corpus."""
    from customer_service_rag.retrieval.indexing import build_index

    settings = load_config(config)
    passages = load_passages(settings.resolve_path(settings.data.passages_path))
    manifest = build_index(
        passages, build_embedding(settings.models.embedding, settings.project_root),
        settings.resolve_path(settings.paths.index_dir),
    )
    emit(manifest)


@app.command()
def retrieve(
    query: str = typer.Option(..., "--query", "-q"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    top_k: Optional[int] = typer.Option(None, "--top-k", min=1),
    full: bool = typer.Option(False, "--full"),
) -> None:
    """Retrieve from the local index and show branch provenance."""
    settings = load_config(config)
    if top_k is not None:
        settings = load_config(config, overrides={"retrieval": {
            "final_top_k": top_k,
            "candidate_top_k": max(settings.retrieval.candidate_top_k, top_k),
        }})
    runner = build_orchestrator(settings)
    plan = QueryPlan(
        conversation_id="cli_retrieval", current_message=query, rewritten_query=query,
    )
    result = asyncio.run(runner.retriever.retrieve(plan, settings.retrieval))
    if full:
        emit(result.model_dump(mode="json"))
    else:
        final_ids = {passage.citation_id for passage in result.passages}
        emit({
            "embedding_backend": result.embedding_backend,
            "elapsed_ms": result.elapsed_ms,
            "passages": [{
                "citation_id": candidate.passage.citation_id,
                "text": candidate.passage.text,
                "fusion_score": candidate.fusion_score,
                "sources": sorted({hit.source for hit in candidate.hits}),
            } for candidate in result.candidates if candidate.passage.citation_id in final_ids],
        })


def _execute_example(
    settings: AppConfig, example: EvaluationExample, save: bool, runner=None,
) -> RunResult:
    # Only the dialogue is sent to the online runner, never references/gold citations.
    result = asyncio.run((runner or build_orchestrator(settings)).run(example.dialogue))
    result.sample_id = example.sample_id
    if save:
        directory = write_run(settings, result)
        typer.echo(f"Run artifacts: {directory}", err=True)
    return result


def result_summary(result: RunResult) -> dict:
    return {
        "run_id": result.run_id, "sample_id": result.sample_id,
        "is_mock": result.is_mock, "mock_scenario": result.mock_scenario,
        "model_backend": result.model_backend,
        "note": (
            "Offline mock decisions demonstrate control flow, not validated answer quality."
            if result.is_mock else "Real local-model development run, not a final benchmark."
        ),
        "response": result.response.model_dump(mode="json"),
        "query": result.query.model_dump(mode="json") if result.query else None,
        "phases": [
            event.phase.value for event in result.trace if event.event == "transition"
        ],
        "feedback": [feedback.model_dump(mode="json") for feedback in result.feedback],
        "actions": [action.model_dump(mode="json") for action in result.actions],
        "usage": [usage.model_dump(mode="json") for usage in result.usage],
        "iteration": result.iteration, "retrieval_round": result.retrieval_round,
        "retry_count": result.retry_count, "stop_reason": result.stop_reason,
        "escalation": result.escalation.model_dump(mode="json") if result.escalation else None,
    }


@app.command()
def scenarios(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """List explicitly scripted D3 mock-service demonstrations."""
    settings = load_config(config)
    catalog = load_scenarios(settings.resolve_path("data/scenarios/d3.yaml"))
    emit([{"name": name, "description": item.description} for name, item in catalog.items()])


def _execute_scenarios(
    name: str, config: Optional[Path], save: bool, full: bool,
) -> list[RunResult]:
    base = load_config(config)
    catalog = load_scenarios(base.resolve_path("data/scenarios/d3.yaml"))
    if name != "all" and name not in catalog:
        raise ValueError(f"Unknown scenario: {name}; use cs-rag scenarios")
    selected = list(catalog.values()) if name == "all" else [catalog[name]]
    config_path = config or base.resolve_path("configs/d3.yaml")
    results = []
    for scenario in selected:
        settings = load_config(config_path, overrides=scenario.config_overrides)
        if not settings.quality.enabled:
            raise ValueError("Scripted scenarios require quality.enabled=true; use configs/d3.yaml")
        evaluator, optimizer, generator = scenario_factories(scenario)
        runner = build_orchestrator(
            settings, evaluator_factory=evaluator, optimizer_factory=optimizer,
            generator_factory=generator, mock_scenario=scenario.name,
        )
        result = asyncio.run(runner.run(scenario.dialogue))
        results.append(result)
        if save:
            directory = write_run(
                settings, result, scenario_definition=scenario.model_dump(mode="json"),
            )
            typer.echo(f"Scripted mock artifacts: {directory}", err=True)
    if full:
        emit([result.model_dump(mode="json") for result in results])
    elif len(results) == 1:
        emit(result_summary(results[0]))
    else:
        emit({
            "is_mock": True, "mode": "scripted_mock_services",
            "note": "These outcomes demonstrate control paths, not research metrics.",
            "runs": [{
                "scenario": result.mock_scenario, "run_id": result.run_id,
                "status": result.response.status.value, "stop_reason": result.stop_reason,
                "evaluations": result.iteration, "retrieval_round": result.retrieval_round,
                "retry_count": result.retry_count,
                "actions": [action.action for action in result.actions],
            } for result in results],
        })
    return results


@app.command()
def demo(
    sample_id: Optional[str] = typer.Option(None, "--sample-id"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    save: bool = typer.Option(True, "--save/--no-save"),
    full: bool = typer.Option(False, "--full"),
    scenario: Optional[str] = typer.Option(None, "--scenario"),
) -> None:
    """Run a mini dialogue or an explicitly scripted D3 scenario (or all)."""
    if scenario is not None:
        if sample_id is not None:
            raise ValueError("Choose --sample-id or --scenario, not both")
        results = _execute_scenarios(scenario, config, save, full)
        if any(result.response.status == ResponseStatus.ERROR for result in results):
            raise typer.Exit(1)
        return
    settings = load_config(config)
    examples = load_examples(settings.resolve_path(settings.data.examples_path))
    sample_id = sample_id or examples[0].sample_id
    example = next((item for item in examples if item.sample_id == sample_id), None)
    if example is None:
        raise ValueError(f"Unknown sample ID: {sample_id}; use cs-rag samples")
    result = _execute_example(settings, example, save)
    if full:
        emit(result.model_dump(mode="json"))
    else:
        emit(result_summary(result))
    if result.response.status == ResponseStatus.ERROR:
        raise typer.Exit(1)


@app.command()
def run(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    split: str = typer.Option("mini", "--split"),
    limit: Optional[int] = typer.Option(None, "--limit", min=1),
    save: bool = typer.Option(True, "--save/--no-save"),
) -> None:
    """Run mini examples sequentially. This does not compute research metrics."""
    settings = load_config(config)
    if settings.data.source == "synthetic" and split != "mini":
        raise ValueError("Synthetic data supports --split mini only")
    if settings.data.source == "multidoc2dial":
        if split == "test":
            raise ValueError("The v1.0 official test is a dummy; use dev or validation explicitly")
        settings = load_config(config, overrides={"data": {
            "split": split,
            "examples_path": str(settings.data.processed_dir / f"{split}.jsonl"),
        }})
    examples = load_examples(settings.resolve_path(settings.data.examples_path))
    selected = [example for example in examples if example.split == split]
    selected = selected[:limit] if limit is not None else selected
    if not selected:
        raise ValueError("No examples selected for the requested split")
    runner = build_orchestrator(settings)
    results = [_execute_example(settings, example, save, runner) for example in selected]
    batch_dir = write_batch(settings, results) if save else None
    emit({
        "is_mock": any(result.is_mock for result in results),
        "batch_dir": str(batch_dir) if batch_dir else None,
        "note": "Development execution; use evaluate for reference-based metrics.",
        "runs": [{
            "sample_id": result.sample_id, "run_id": result.run_id,
            "status": result.response.status.value, "stop_reason": result.stop_reason,
            "evaluations": result.iteration, "retrieval_round": result.retrieval_round,
            "retry_count": result.retry_count,
        } for result in results],
    })
    if any(result.response.status == ResponseStatus.ERROR for result in results):
        raise typer.Exit(1)


@app.command()
def evaluate(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    examples: Optional[Path] = typer.Option(None, "--examples", exists=True, dir_okay=False),
    annotations: Optional[Path] = typer.Option(None, "--annotations", exists=True, dir_okay=False),
    include_mock: bool = typer.Option(False, "--include-mock"),
) -> None:
    """Compute retrieval, answer, citation and efficiency metrics for a saved run/batch."""
    from customer_service_rag.evaluation.runner import evaluate_saved
    if config is None and (run_dir / "config.yaml").exists():
        config = run_dir / "config.yaml"
    settings = load_config(config)
    path = examples or settings.resolve_path(settings.data.examples_path)
    emit(evaluate_saved(
        run_dir, path, annotations_path=annotations, include_mock=include_mock,
    ))


def main() -> None:
    try:
        app()
    except (OSError, ValueError, NotImplementedError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc
