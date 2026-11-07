"""Run a reproducible, resumable random validation sample through Generator."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from customer_service_rag.config import load_config
from customer_service_rag.data.io import load_examples
from customer_service_rag.observability.tracing import write_run
from customer_service_rag.orchestration.orchestrator import build_orchestrator
from customer_service_rag.schemas import ResponseStatus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/validation_a_seed42_n1000"))
    args = parser.parse_args()

    settings = load_config("configs/integrated_single.yaml", overrides={
        "data": {"split": "validation", "examples_path":
                 "data/processed/multidoc2dial/validation.jsonl"},
    })
    examples = load_examples(settings.resolve_path(settings.data.examples_path))
    if not 0 < args.count <= len(examples):
        raise ValueError("count must be within the validation split")
    selected_ids = set(random.Random(args.seed).sample(
        [example.sample_id for example in examples], args.count,
    ))
    selected = [example for example in examples if example.sample_id in selected_ids]
    output = settings.resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selection_path = output / "selection.json"
    selection = {"source": "official MultiDoc2Dial validation", "seed": args.seed,
                 "population": len(examples), "count": args.count,
                 "sample_ids": [example.sample_id for example in selected]}
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
            raise ValueError("Existing selection differs from the requested sample")
    else:
        selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")

    progress_path = output / "progress.jsonl"
    completed = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["status"] != ResponseStatus.ERROR.value:
                completed[record["sample_id"]] = record

    runner = build_orchestrator(settings)
    consecutive_failed_samples = 0
    for position, example in enumerate(selected, 1):
        if example.sample_id in completed:
            continue
        for attempt in range(1, args.max_attempts + 1):
            result = asyncio.run(runner.run(example.dialogue))
            result.sample_id = example.sample_id
            saved = write_run(settings, result)
            record = {"position": position, "sample_id": example.sample_id,
                      "run_id": result.run_id, "run_dir": str(saved),
                      "attempt": attempt, "status": result.response.status.value,
                      "stop_reason": result.stop_reason}
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{position}/{args.count} attempt={attempt} "
                  f"status={record['status']} sample_id={example.sample_id}", flush=True)
            if result.response.status != ResponseStatus.ERROR:
                completed[example.sample_id] = record
                consecutive_failed_samples = 0
                break
            if attempt < args.max_attempts:
                time.sleep(min(30, 2 ** attempt))
        else:
            consecutive_failed_samples += 1
            if consecutive_failed_samples >= 3:
                print("Stopped after three consecutive failed samples; resume when the "
                      "model service is healthy.", flush=True)
                break

    remaining = args.count - len(completed)
    summary = {"selected": args.count, "completed": len(completed),
               "remaining": remaining, "selection": str(selection_path),
               "progress": str(progress_path)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    if remaining:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
