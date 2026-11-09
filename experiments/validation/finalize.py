"""Summarize final predictions and the cost of every saved validation attempt."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np

from experiments import ROOT
from customer_service_rag.config import load_config
from customer_service_rag.data.io import load_examples
from customer_service_rag.evaluation.grounding_metrics import citation_scores
from customer_service_rag.evaluation.response_metrics import response_scores
from customer_service_rag.evaluation.retrieval_metrics import retrieval_scores
from customer_service_rag.schemas import ResponseStatus, RunResult


def avg(values):
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def run_path(record, settings):
    if record.get("run_dir"):
        path = Path(record["run_dir"]) / "result.json"
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            return path
    return settings.resolve_path(settings.paths.results_dir) / record["run_id"] / "result.json"


def usage_summary(result):
    totals = [u.total_tokens for u in result.usage]
    return {
        "latency_ms": result.elapsed_ms,
        "model_calls": len(result.model_calls) or len(result.usage),
        "known_total_tokens": sum(t for t in totals if t is not None),
        "total_tokens": None if any(t is None for t in totals) else sum(totals),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/validation_a_seed42_n1000")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/deepseek.yaml")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    output_dir = (args.output_dir or root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    attempts = [json.loads(line) for line in (root / "progress.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    latest, first = {}, {}
    for item in attempts:
        latest[item["sample_id"]] = item
        first.setdefault(item["sample_id"], item)
    ids = selection["sample_ids"]
    if not ids or len(ids) != selection["count"] or set(ids) != set(latest):
        raise ValueError("Every selected sample must have a saved attempt")
    settings = load_config(args.config, overrides={
        "quality": {"enabled": False}, "optimization": {"enabled": False},
        "escalation": {"enabled": False}, "data": {
            "split": "validation", "examples_path": "data/processed/multidoc2dial/validation.jsonl"},
    })
    examples = {item.sample_id: item for item in load_examples(
        settings.resolve_path(settings.data.examples_path)) if item.sample_id in latest}
    saved = {}
    for record in attempts:
        path = run_path(record, settings)
        result = RunResult.model_validate_json(path.read_text(encoding="utf-8"))
        if result.sample_id != record["sample_id"] or result.is_mock or result.iteration != 0:
            raise ValueError(f"Invalid single-pass result: {record['sample_id']}")
        saved[record["run_id"]] = result
    all_costs = [usage_summary(saved[record["run_id"]]) for record in attempts]
    last_costs = [usage_summary(saved[latest[sid]["run_id"]]) for sid in ids]
    rows, counts = [], Counter()
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for sid in ids:
            record = latest[sid]
            result = saved[record["run_id"]]
            example = examples[sid]
            status = result.response.status.value
            counts[status] += 1
            stream.write(json.dumps({
                "sample_id": sid, "run_id": result.run_id, "status": status,
                "answer": result.response.answer, "citations": result.response.citations,
                "stop_reason": result.stop_reason, "run_file": str(run_path(record, settings)),
            }, ensure_ascii=False) + "\n")
            rows.append({"sample_id": sid, "status": status,
                         "response": response_scores(result.response.answer, example.reference_answers,
                                                     result.response.status == ResponseStatus.ANSWERED),
                         "retrieval": retrieval_scores(result.retrieval, example),
                         "citations": citation_scores(result, example)})
    unknown = sum(cost["total_tokens"] is None for cost in all_costs)
    metrics = {
        "sample_count": len(ids), "status_counts": dict(counts),
        "response_f1": avg(row["response"]["f1"] for row in rows),
        "response_em": avg(row["response"]["em"] for row in rows),
        "retrieval": {f"@{k}": {metric: avg(row["retrieval"][f"@{k}"][metric] for row in rows)
                                   for metric in ("recall", "hit", "complete_coverage")}
                      for k in (1, 5, 10)},
        "citation_evidence_validity": avg(row["citations"]["evidence_validity"] for row in rows),
        "citation_gold_precision": avg(row["citations"]["gold_precision"] for row in rows),
        "last_attempt": {
            "average_latency_ms": avg(c["latency_ms"] for c in last_costs),
            "p95_latency_ms": float(np.percentile([c["latency_ms"] for c in last_costs], 95)),
            "average_model_calls": avg(c["model_calls"] for c in last_costs),
            "average_total_tokens_when_known": avg(c["total_tokens"] for c in last_costs),
            "unknown_usage_attempts": sum(c["total_tokens"] is None for c in last_costs),
        },
        "all_attempts": {
            "attempt_count": len(attempts),
            "retried_samples": sum(n > 1 for n in Counter(r["sample_id"] for r in attempts).values()),
            "first_attempt_status_counts": dict(Counter(record["status"] for record in first.values())),
            "total_model_calls": sum(c["model_calls"] for c in all_costs),
            "total_run_latency_ms": sum(c["latency_ms"] for c in all_costs),
            "average_run_latency_ms_per_sample": sum(c["latency_ms"] for c in all_costs) / len(ids),
            "known_total_tokens": sum(c["known_total_tokens"] for c in all_costs),
            "unknown_usage_attempts": unknown,
            "total_tokens": None if unknown else sum(c["total_tokens"] for c in all_costs),
            "timing_note": "Sum of recorded request runtimes; excludes retry backoff, model loading and indexing.",
        },
        "hallucination_rate": None, "grounding_accuracy": None,
        "dialogue_consistency": None, "task_completion_rate": None,
        "notes": ["Final errors and insufficient-evidence responses score zero against a reference.",
                  "Cost includes all saved attempts; unknown provider usage remains unknown.",
                  "Factual support and task completion require independent annotations."],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "status", "f1", "em", "recall@1", "recall@5", "recall@10"])
        for row in rows:
            writer.writerow([row["sample_id"], row["status"], row["response"]["f1"], row["response"]["em"],
                             *[row["retrieval"][f"@{k}"]["recall"] for k in (1, 5, 10)]])
    print(json.dumps({"sample_count": len(ids), "status_counts": dict(counts),
                      "attempt_count": len(attempts), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
