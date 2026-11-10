"""Stream the saved full-validation runs into a compact, reproducible report."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from urllib.parse import unquote

import numpy as np

from customer_service_rag.data.io import load_examples
from customer_service_rag.evaluation.grounding_metrics import citation_scores
from customer_service_rag.evaluation.response_metrics import response_scores
from customer_service_rag.evaluation.retrieval_metrics import retrieval_scores
from customer_service_rag.retrieval.corpus import load_passages_jsonl, passage_fingerprint
from customer_service_rag.schemas import ResponseStatus, RunResult
from experiments import ROOT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average(values):
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/validation_native_full")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/validation_native_full")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    selection_path = run_dir / "selection.json"
    progress_path = run_dir / "progress.jsonl"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    sample_ids = selection["sample_ids"]
    if len(sample_ids) != 4427 or len(set(sample_ids)) != 4427:
        parser.error("Expected 4,427 unique official validation samples")
    records = {}
    with progress_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                records[record["sample_id"]] = record
    if set(records) != set(sample_ids):
        parser.error("Progress does not contain a final attempt for every selected sample")

    examples_path = ROOT / "data/processed/multidoc2dial/validation.jsonl"
    corpus_path = ROOT / "artifacts/indexes/struct/finetuned/passages.jsonl"
    if sha256(examples_path) != selection["examples_sha256"]:
        parser.error("Validation data differs from the selected run")
    examples = {item.sample_id: item for item in load_examples(examples_path)}
    corpus = load_passages_jsonl(corpus_path)
    fingerprint = passage_fingerprint(corpus)
    spans = {}
    for passage in corpus:
        for span_id in passage.span_ids:
            key = passage.domain, passage.doc_id, str(span_id)
            if key in spans:
                parser.error(f"Duplicate corpus span: {key}")
            spans[key] = passage.passage_id
    for example in examples.values():
        gold = set()
        for reference in example.grounding:
            domain = (unquote(reference.doc_id.split("/", 1)[0])
                      if "/" in reference.doc_id else example.domain)
            doc_id = reference.original_doc_id or unquote(reference.doc_id.split("/", 1)[-1])
            for span_id in reference.span_ids:
                gold.add(spans[(domain, doc_id, str(span_id))])
        example.gold_citations = sorted(gold)
        example.gold_evidence_groups = [[item] for item in sorted(gold)]

    output.mkdir(parents=True, exist_ok=True)
    rows, latencies, tokens, rounds, retries, calls = [], [], [], [], [], []
    statuses = Counter()
    token_sources = set()
    evaluation_path = output / "evaluation.jsonl"
    with evaluation_path.open("w", encoding="utf-8", newline="\n") as stream:
        for sample_id in sample_ids:
            record = records[sample_id]
            result_path = Path(record["run_dir"]) / "result.json"
            result = RunResult.model_validate_json(result_path.read_text(encoding="utf-8"))
            if result.sample_id != sample_id or result.response.status.value != record["status"]:
                parser.error(f"Progress/result mismatch: {sample_id}")
            if result.is_mock or result.integration.get("citation_scheme") != "native_doc::struct":
                parser.error(f"Unexpected model or citation scheme: {sample_id}")
            saved_corpus = result.integration.get("corpus_version")
            if saved_corpus != fingerprint:
                parser.error(f"Corpus mismatch: {sample_id}")
            example = examples[sample_id]
            row = {
                "sample_id": sample_id, "status": record["status"],
                "stop_reason": record["stop_reason"],
                "response": response_scores(result.response.answer, example.reference_answers,
                                            result.response.status == ResponseStatus.ANSWERED),
                "retrieval": retrieval_scores(result.retrieval, example),
                "citations": citation_scores(result, example),
                "elapsed_ms": result.elapsed_ms, "retry_count": result.retry_count,
                "model_calls": len(result.model_calls) or len(result.usage),
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            statuses[record["status"]] += 1
            latencies.append(result.elapsed_ms)
            rounds.append(result.retrieval_round)
            retries.append(result.retry_count)
            calls.append(row["model_calls"])
            tokens.append(sum(item.total_tokens for item in result.usage)
                          if all(item.total_tokens is not None for item in result.usage) else None)
            token_sources.update(item.source for item in result.usage)

    metrics = {
        "sample_count": len(rows), "status_counts": dict(statuses),
        "response_f1": average(row["response"]["f1"] for row in rows),
        "response_em": average(row["response"]["em"] for row in rows),
        "retrieval": {f"@{k}": {metric: average(row["retrieval"][f"@{k}"][metric]
                                         for row in rows)
                                  for metric in ("recall", "hit", "complete_coverage")}
                      for k in (1, 5, 10)},
        "retrieval_scored_count": sum(row["retrieval"]["@1"]["recall"] is not None for row in rows),
        "citation_evidence_validity": average(row["citations"]["evidence_validity"] for row in rows),
        "citation_gold_precision": average(row["citations"]["gold_precision"] for row in rows),
        "average_latency_ms": mean(latencies), "p95_latency_ms": float(np.percentile(latencies, 95)),
        "average_retrieval_rounds": mean(rounds), "average_retry_count": mean(retries),
        "average_model_calls": mean(calls), "average_total_tokens": average(tokens),
        "complete_usage_count": sum(value is not None for value in tokens),
        "token_sources": sorted(token_sources),
        "note": "All 4,427 requests are in the denominator; error and other unanswered statuses score F1/EM zero.",
    }
    manifest = {
        "dataset": "MultiDoc2Dial official validation", "sample_count": len(sample_ids),
        "examples_sha256": sha256(examples_path), "corpus_sha256": sha256(corpus_path),
        "corpus_version": fingerprint, "selection_sha256": sha256(selection_path),
        "progress_sha256": sha256(progress_path), "config_sha256": selection["config_sha256"],
        "retrieval": "struct fine-tuned BGE Dense, full corpus, top 10, no reranker",
        "workflow": "planner -> retrieval -> DeepSeek generation -> evaluator -> optimizer",
        "error_policy": "last saved error kept; F1/EM zero; no fabricated clarification",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
