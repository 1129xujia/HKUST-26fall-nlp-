"""Create compact predictions and metrics from saved validation A runs."""

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np

from customer_service_rag.config import load_config
from customer_service_rag.data.io import load_examples
from customer_service_rag.evaluation.grounding_metrics import citation_scores
from customer_service_rag.evaluation.response_metrics import response_scores
from customer_service_rag.evaluation.retrieval_metrics import retrieval_scores
from customer_service_rag.schemas import ResponseStatus, RunResult


def avg(values):
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def main():
    root = Path("results/validation_a_seed42_n1000")
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    attempts = [json.loads(line) for line in (root / "progress.jsonl").read_text(
        encoding="utf-8").splitlines()]
    latest = {item["sample_id"]: item for item in attempts}
    ids = selection["sample_ids"]
    if len(ids) != selection["count"] or set(ids) != set(latest):
        raise ValueError("The selected 1,000 samples were not all attempted")
    settings = load_config("configs/integrated_single.yaml", overrides={"data": {
        "split": "validation", "examples_path": "data/processed/multidoc2dial/validation.jsonl"}})
    examples = {item.sample_id: item for item in load_examples(
        settings.resolve_path(settings.data.examples_path)) if item.sample_id in latest}
    rows, latencies, token_totals, call_counts = [], [], [], []
    counts = Counter()
    with (root / "predictions.jsonl").open("w", encoding="utf-8") as output:
        for sample_id in ids:
            record = latest[sample_id]
            path = Path("results") / record["run_id"] / "result.json"
            result = RunResult.model_validate_json(path.read_text(encoding="utf-8"))
            if result.sample_id != sample_id or result.is_mock or result.iteration != 0:
                raise ValueError(f"Invalid A result: {sample_id}")
            status = result.response.status.value
            counts[status] += 1
            latencies.append(result.elapsed_ms)
            call_counts.append(len(result.model_calls) or len(result.usage))
            token_totals.append(sum(u.total_tokens for u in result.usage)
                                if all(u.total_tokens is not None for u in result.usage)
                                else None)
            output.write(json.dumps({"sample_id": sample_id, "run_id": result.run_id,
                                     "status": status, "answer": result.response.answer,
                                     "citations": result.response.citations,
                                     "stop_reason": result.stop_reason,
                                     "run_file": str(path)}, ensure_ascii=False) + "\n")
            example = examples[sample_id]
            rows.append({"sample_id": sample_id, "status": status,
                         "response": response_scores(result.response.answer,
                             example.reference_answers,
                             result.response.status == ResponseStatus.ANSWERED),
                         "retrieval": retrieval_scores(result.retrieval, example),
                         "citations": citation_scores(result, example)})
    metrics = {
        "sample_count": len(ids), "status_counts": dict(counts),
        "response_f1": avg(row["response"]["f1"] for row in rows),
        "response_em": avg(row["response"]["em"] for row in rows),
        "retrieval": {f"@{k}": {metric: avg(row["retrieval"][f"@{k}"][metric]
                                       for row in rows)
                                for metric in ("recall", "hit", "complete_coverage")}
                      for k in (1, 5, 10)},
        "retrieval_scored_count": sum(row["retrieval"]["@1"]["recall"] is not None
                                      for row in rows),
        "citation_evidence_validity": avg(row["citations"]["evidence_validity"]
                                          for row in rows),
        "citation_gold_precision": avg(row["citations"]["gold_precision"]
                                       for row in rows),
        "average_latency_ms": mean(latencies),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "average_model_calls": mean(call_counts),
        "average_total_tokens": avg(token_totals),
        "complete_usage_count": sum(value is not None for value in token_totals),
        "hallucination_rate": None, "grounding_accuracy": None,
        "dialogue_consistency": None, "task_completion_rate": None,
        "notes": ["Error and insufficient-evidence responses score zero against a reference.",
                  "Factual support and task completion need external annotations."],
    }
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "status", "f1", "em", "recall@1", "recall@5", "recall@10"])
        for row in rows:
            writer.writerow([row["sample_id"], row["status"], row["response"]["f1"],
                             row["response"]["em"], *[
                                 row["retrieval"][f"@{k}"]["recall"] for k in (1, 5, 10)]])
    summary = {"selected": len(ids), "processed_unique": len(latest),
               "attempts": len(attempts), "status_counts": dict(counts),
               "seed": selection["seed"], "source_commit": "ccf5933",
               "selection": "selection.json", "progress": "progress.jsonl",
               "predictions": "predictions.jsonl", "metrics": "metrics.json"}
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "response_f1": metrics["response_f1"]}, indent=2))


if __name__ == "__main__":
    main()
