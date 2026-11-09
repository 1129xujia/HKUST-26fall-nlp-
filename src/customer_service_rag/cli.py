"""One CLI for data, indexing, retrieval, answers and offline evaluation."""

from __future__ import annotations

import asyncio
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Optional
from uuid import uuid4

import typer
import yaml
from pydantic.json_schema import models_json_schema

from customer_service_rag import __version__, schemas
from customer_service_rag.config import AppConfig, load_config
from customer_service_rag.data.io import load_examples, load_passages
from customer_service_rag.data.preprocessing import prepare_mini
from customer_service_rag.observability.tracing import write_batch, write_run
from customer_service_rag.orchestration.orchestrator import build_orchestrator
from customer_service_rag.orchestration.retrieval import build_index_for_config, describe_assets
from customer_service_rag.schemas import DialogueInput, QueryPlan, ResponseStatus, RunResult

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False,
                  help="Customer-service RAG: team retrieval, cited generation and bounded optimization.")


def emit(value: object) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2))


@app.command("config")
def show_config(config: Optional[Path] = typer.Option(None, "--config", "-c"),
                experiment: Optional[str] = typer.Option(None, "--experiment")) -> None:
    """Print the fully expanded configuration."""
    typer.echo(yaml.safe_dump(load_config(config, experiment=experiment).model_dump(mode="json"),
                              sort_keys=False))


@app.command()
def doctor(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Describe local assets and dependencies without loading models."""
    settings = load_config(config)
    passages = settings.resolve_path(settings.data.passages_path)
    examples = settings.resolve_path(settings.data.examples_path)
    if settings.retrieval.backend == "native":
        from customer_service_rag.retrieval.corpus import load_passages_jsonl

        passages = settings.resolve_path(settings.paths.index_dir) / "passages.jsonl"
        passage_count = len(load_passages_jsonl(passages)) if passages.is_file() else None
    else:
        passage_count = len(load_passages(passages)) if passages.is_file() else None
    emit({
        "package_version": __version__, "python": platform.python_version(),
        "project_root": str(settings.project_root),
        "llm_backend": settings.models.llm.backend, "llm_model": settings.models.llm.model,
        "api_key_configured": bool(os.environ.get("DEEPSEEK_API_KEY", "").strip()),
        "quality_enabled": settings.quality.enabled,
        "optimization_enabled": settings.optimization.enabled,
        "data_source": settings.data.source,
        "passages": passage_count,
        "examples": len(load_examples(examples)) if examples.is_file() else None,
        "retrieval": describe_assets(settings),
        "dependencies": {name: version(name) for name in ("pydantic", "PyYAML", "numpy", "rank-bm25")},
    })


@app.command("schema")
def export_schema(output: Optional[Path] = typer.Option(None, "--output", "-o")) -> None:
    """Export the application and optimizer contracts on demand."""
    from customer_service_rag.optimization.schemas import Feedback, OptimizationInput, OptimizationResult
    contracts = [schemas.DialogueInput, schemas.QueryPlan, schemas.Passage,
                 schemas.GeneratedResponse, schemas.RunResult, schemas.EvaluationExample,
                 AppConfig, Feedback, OptimizationInput, OptimizationResult]
    _, document = models_json_schema([(model, "validation") for model in contracts],
                                     title="Customer Service RAG")
    document["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    if output is None:
        emit(document)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
        typer.echo(str(output.resolve()))


@app.command("download-data")
def download_data(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Download official MultiDoc2Dial data with source checksums."""
    from customer_service_rag.assets import download_dataset
    manifest = download_dataset(load_config(config).project_root)
    emit({key: value for key, value in manifest.items() if key != "files"})


@app.command("download-models")
def download_model_assets(endpoint: str = typer.Option("https://hf-mirror.com", "--endpoint"),
                          config: Optional[Path] = typer.Option(None, "--config", "-c"),
                          include_llm: bool = typer.Option(False, "--include-llm")) -> None:
    """Download pinned shared embeddings/reranker; optionally the local LLM."""
    from customer_service_rag.assets import download_models
    registry = download_models(load_config(config).project_root, endpoint, include_llm=include_llm)
    emit({key: {name: entry[name] for name in ("repo_id", "revision", "path", "license")}
          for key, entry in registry.items()})


@app.command()
def preprocess(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Prepare examples while keeping references outside online model inputs."""
    settings = load_config(config)
    if settings.data.source == "multidoc2dial":
        from customer_service_rag.data.official import prepare_official
        emit(prepare_official(settings))
    else:
        emit(prepare_mini(settings))


@app.command()
def index(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Build the configured shared NumPy or native FAISS index."""
    emit(build_index_for_config(load_config(config)))


@app.command()
def retrieve(query: str = typer.Option(..., "--query", "-q"),
             config: Optional[Path] = typer.Option(None, "--config", "-c"),
             top_k: Optional[int] = typer.Option(None, "--top-k", min=1, max=20),
             full: bool = typer.Option(False, "--full")) -> None:
    """Inspect retrieval without making an LLM request."""
    settings = load_config(config)
    if top_k is not None:
        settings.retrieval = settings.retrieval.model_copy(update={
            "final_top_k": top_k,
            "candidate_top_k": max(settings.retrieval.candidate_top_k, top_k),
        })
    retriever = build_orchestrator(settings).retriever
    dialogue = DialogueInput(conversation_id="retrieval", current_message=query)
    plan = retriever.plan_for_retrieval(QueryPlan(**dialogue.model_dump(), rewritten_query=query), dialogue)
    result = asyncio.run(retriever.retrieve(plan, settings.retrieval))
    emit(result.model_dump(mode="json") if full else {
        "embedding_backend": result.embedding_backend, "elapsed_ms": result.elapsed_ms,
        "passages": [{"citation_id": p.citation_id, "title": p.title, "text": p.text}
                     for p in result.passages],
    })


def result_summary(result: RunResult) -> dict:
    return {
        "run_id": result.run_id, "sample_id": result.sample_id, "is_mock": result.is_mock,
        "response": result.response.model_dump(mode="json"),
        "retrieval_rounds": result.retrieval_round, "evaluations": result.iteration,
        "retries": result.retry_count, "stop_reason": result.stop_reason,
        "escalation": {"reason": result.escalation.reason,
                       "delivered_to_human": result.escalation.delivered_to_human}
        if result.escalation else None,
    }


@app.command()
def ask(message: str = typer.Option(..., "--message", "-m"),
        config: Optional[Path] = typer.Option(None, "--config", "-c"),
        experiment: Optional[str] = typer.Option(None, "--experiment"),
        history: Optional[Path] = typer.Option(None, "--history", exists=True, dir_okay=False),
        save: bool = typer.Option(True, "--save/--no-save"),
        full: bool = typer.Option(False, "--full")) -> None:
    """Answer a message with optional visible role/text history."""
    settings = load_config(config, experiment=experiment)
    dialogue = DialogueInput(conversation_id=uuid4().hex, current_message=message,
        dialogue_history=json.loads(history.read_text(encoding="utf-8")) if history else [])
    result = asyncio.run(build_orchestrator(settings).run(dialogue))
    if save:
        typer.echo(f"Run artifacts: {write_run(settings, result)}", err=True)
    emit(result.model_dump(mode="json") if full else result_summary(result))
    if result.response.status == ResponseStatus.ERROR:
        raise typer.Exit(1)


@app.command()
def run(config: Optional[Path] = typer.Option(None, "--config", "-c"),
        experiment: Optional[str] = typer.Option(None, "--experiment"),
        split: Optional[str] = typer.Option(None, "--split"),
        limit: Optional[int] = typer.Option(None, "--limit", min=1),
        save: bool = typer.Option(True, "--save/--no-save")) -> None:
    """Run a dataset split through the same pipeline as ask."""
    settings = load_config(config, experiment=experiment)
    split = split or settings.data.split
    if settings.data.source == "synthetic" and split != "mini":
        raise ValueError("Synthetic examples support --split mini only")
    if settings.data.source == "multidoc2dial":
        if split not in {"mini", "train", "dev", "validation"}:
            raise ValueError("Use mini/train/dev/validation; the official v1.0 test is dummy")
        settings.data = settings.data.model_copy(update={
            "split": split, "examples_path": settings.data.processed_dir / f"{split}.jsonl",
        })
    examples = [e for e in load_examples(settings.resolve_path(settings.data.examples_path)) if e.split == split]
    if limit is not None:
        examples = examples[:limit]
    if not examples:
        raise ValueError("No examples selected")
    runner = build_orchestrator(settings)
    results = []
    for example in examples:
        result = asyncio.run(runner.run(example.dialogue))
        result.sample_id = example.sample_id
        results.append(result)
        if save:
            write_run(settings, result)
    directory = write_batch(settings, results) if save else None
    emit({"batch_dir": str(directory) if directory else None,
          "runs": [{"run_id": r.run_id, "sample_id": r.sample_id,
                    "status": r.response.status.value, "stop_reason": r.stop_reason,
                    "retrieval_rounds": r.retrieval_round, "evaluations": r.iteration,
                    "retries": r.retry_count} for r in results]})
    if any(r.response.status == ResponseStatus.ERROR for r in results):
        raise typer.Exit(1)


@app.command()
def evaluate(run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False),
             config: Optional[Path] = typer.Option(None, "--config", "-c"),
             examples: Optional[Path] = typer.Option(None, "--examples", exists=True, dir_okay=False),
             annotations: Optional[Path] = typer.Option(None, "--annotations", exists=True, dir_okay=False),
             include_mock: bool = typer.Option(False, "--include-mock")) -> None:
    """Score saved results with held-out references and optional support annotations."""
    from customer_service_rag.evaluation.runner import evaluate_saved
    if config is None and (run_dir / "config.yaml").is_file():
        config = run_dir / "config.yaml"
    settings = load_config(config)
    emit(evaluate_saved(run_dir, examples or settings.resolve_path(settings.data.examples_path),
        annotations_path=annotations, include_mock=include_mock,
        native_corpus_path=(settings.resolve_path(settings.paths.index_dir) / "passages.jsonl")
        if settings.retrieval.backend == "native" else None))


def main() -> None:
    try:
        app()
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc
