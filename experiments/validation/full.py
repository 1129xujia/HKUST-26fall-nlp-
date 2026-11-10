"""Resumable full-pipeline run over every official validation request."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from customer_service_rag.config import load_config
from customer_service_rag.data.io import load_examples
from customer_service_rag.observability.tracing import write_run
from customer_service_rag.orchestration.orchestrator import build_orchestrator
from customer_service_rag.schemas import ResponseStatus
from experiments import ROOT


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/native.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/validation_native_full")
    parser.add_argument("--limit", type=int, help="Small wiring check; omit for all validation")
    parser.add_argument("--mock", action="store_true", help="Offline wiring check; use a separate output directory")
    parser.add_argument("--max-attempts", type=int, default=4)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    overrides = {"data": {
        "split": "validation", "examples_path": "data/processed/multidoc2dial/validation.jsonl",
    }}
    if args.mock:
        overrides["models"] = {"llm": {"backend": "mock", "model": "mock-extractive-v1"}}
    settings = load_config(args.config, overrides=overrides)
    if (settings.retrieval.backend, settings.retrieval.segmentation, settings.retrieval.chain) != (
        "native", "struct", "finetuned"
    ) or not (settings.quality.enabled and settings.optimization.enabled):
        parser.error("Use a native struct + finetuned config with quality and optimization enabled")
    examples_path = settings.resolve_path(settings.data.examples_path)
    examples = [row for row in load_examples(examples_path) if row.split == "validation"]
    if args.limit is not None:
        examples = examples[:args.limit]
    if not examples:
        parser.error("No validation examples found")
    output = settings.resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selection = {
        "source": "official MultiDoc2Dial validation", "count": len(examples),
        "sample_ids": [row.sample_id for row in examples],
        "examples_sha256": sha256(examples_path), "config_sha256": sha256(args.config.resolve()),
        "backend": "native", "segmentation": "struct", "chain": "finetuned",
        "index_dir": str(settings.resolve_path(settings.paths.index_dir)),
        "llm_backend": settings.models.llm.backend,
    }
    selection_path = output / "selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
            parser.error("Existing output uses different samples or config; choose another --output")
    else:
        selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")

    progress_path = output / "progress.jsonl"
    completed = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record["status"] != ResponseStatus.ERROR.value:
                    completed[record["sample_id"]] = record
    runner = build_orchestrator(settings)
    failed_in_a_row = 0
    for position, example in enumerate(examples, 1):
        if example.sample_id in completed:
            continue
        for attempt in range(1, args.max_attempts + 1):
            result = asyncio.run(runner.run(example.dialogue))
            result.sample_id = example.sample_id
            saved = write_run(settings, result)
            record = {
                "position": position, "sample_id": example.sample_id, "run_id": result.run_id,
                "run_dir": str(saved), "attempt": attempt,
                "status": result.response.status.value, "stop_reason": result.stop_reason,
            }
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{position}/{len(examples)} attempt={attempt} "
                  f"status={record['status']} sample_id={example.sample_id}", flush=True)
            if result.response.status != ResponseStatus.ERROR:
                completed[example.sample_id] = record
                failed_in_a_row = 0
                break
            if attempt < args.max_attempts:
                time.sleep(min(30, 2 ** attempt))
        else:
            failed_in_a_row += 1
            if failed_in_a_row >= 3:
                print("Stopped after three consecutive failed samples; rerun the same command later.", flush=True)
                break

    summary = {
        "selected": len(examples), "completed": len(completed),
        "remaining": len(examples) - len(completed),
        "status_counts": dict(Counter(item["status"] for item in completed.values())),
        "selection": str(selection_path), "progress": str(progress_path),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["remaining"]:
        raise SystemExit(2)

    # The standard offline evaluator expects full RunResult JSONL records.
    predictions = output / "predictions.jsonl"
    with predictions.open("w", encoding="utf-8", newline="\n") as stream:
        for example in examples:
            record = completed[example.sample_id]
            path = Path(record["run_dir"]) / "result.json"
            if not path.is_absolute():
                path = ROOT / path
            stream.write(json.dumps(json.loads(path.read_text(encoding="utf-8")),
                                    ensure_ascii=False) + "\n")
    print(f"Saved complete batch for offline evaluation: {predictions}")


if __name__ == "__main__":
    main()
