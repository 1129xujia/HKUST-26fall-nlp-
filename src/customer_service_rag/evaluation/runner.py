"""Evaluate saved predictions against held-out references without feeding them online."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from urllib.parse import unquote

import numpy as np

from customer_service_rag.data.io import load_examples, read_jsonl
from customer_service_rag.evaluation.grounding_metrics import citation_scores, factual_support_scores
from customer_service_rag.evaluation.response_metrics import response_scores
from customer_service_rag.evaluation.retrieval_metrics import retrieval_scores
from customer_service_rag.schemas import ResponseStatus, RunResult


def _average(values) -> float | None:
    valid = [value for value in values if value is not None]
    return mean(valid) if valid else None


def evaluate_saved(
    directory: Path, examples_path: Path, *, annotations_path: Path | None = None,
    include_mock: bool = False, native_corpus_path: Path | None = None,
) -> dict:
    if (directory / "result.json").is_file():
        results = [RunResult.model_validate_json((directory / "result.json").read_text(encoding="utf-8"))]
    elif (directory / "predictions.jsonl").is_file():
        results = read_jsonl(directory / "predictions.jsonl", RunResult)
    else:
        raise ValueError("Select a single run directory or a batch with predictions.jsonl")
    if not results:
        raise ValueError("No predictions to evaluate")
    if any(result.is_mock or result.mock_scenario for result in results) and not include_mock:
        raise ValueError("Mock runs are excluded by default; use --include-mock only for development")
    ids = [result.sample_id for result in results]
    if None in ids or len(ids) != len(set(ids)):
        raise ValueError("Predictions require unique, non-null sample_id values")
    examples = {example.sample_id: example for example in load_examples(examples_path)}
    if not set(ids).issubset(examples):
        raise ValueError("Some prediction IDs do not occur in the selected reference file")
    native = [r.integration.get("citation_scheme", "").startswith("native_doc::") for r in results]
    if any(native):
        schemes = {r.integration.get("citation_scheme") for r in results}
        if not all(native) or len(schemes) != 1 or native_corpus_path is None:
            raise ValueError("Native evaluation needs one matching native corpus and citation scheme")
        from customer_service_rag.retrieval.corpus import load_passages_jsonl, passage_fingerprint

        corpus = load_passages_jsonl(native_corpus_path)
        fingerprint = passage_fingerprint(corpus)
        span_index = {}
        for passage in corpus:
            for span_id in passage.span_ids:
                key = (passage.domain, passage.doc_id, str(span_id))
                if key in span_index:
                    raise ValueError("Native corpus maps a span to multiple passages")
                span_index[key] = passage.passage_id
        for result in results:
            saved_corpus = result.integration.get("corpus", {})
            expected = (result.integration.get("corpus_version")
                        or saved_corpus.get("corpus_metadata", {}).get("passage_fingerprint")
                        or saved_corpus.get("dense_metadata", {}).get("passage_fingerprint"))
            if expected != fingerprint:
                raise ValueError("The native evaluation corpus differs from the run corpus")
            example = examples[result.sample_id]
            gold = set()
            for reference in example.grounding:
                domain = (unquote(reference.doc_id.split("/", 1)[0])
                          if "/" in reference.doc_id else example.domain)
                doc_id = reference.original_doc_id or unquote(reference.doc_id.split("/", 1)[-1])
                for span_id in reference.span_ids:
                    key = (domain, doc_id, str(span_id))
                    if key not in span_index:
                        raise ValueError(f"Unmapped native grounding reference: {key}")
                    gold.add(span_index[key])
            if not example.grounding and example.gold_citations:
                raise ValueError("Native gold projection requires original grounding spans")
            examples[result.sample_id] = example.model_copy(update={
                "gold_citations": sorted(gold),
                "gold_evidence_groups": [[citation] for citation in sorted(gold)],
            })
    annotations = {}
    if annotations_path:
        for line in annotations_path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                item = json.loads(line)
                if item["sample_id"] in annotations:
                    raise ValueError("Duplicate sample in support annotations")
                annotations[item["sample_id"]] = item
    rows = []
    for result in results:
        example = examples[result.sample_id]
        row = {
            "sample_id": result.sample_id, "run_id": result.run_id,
            "status": result.response.status.value, "stop_reason": result.stop_reason,
            "response": response_scores(
                result.response.answer, example.reference_answers,
                result.response.status == ResponseStatus.ANSWERED,
            ),
            "retrieval": retrieval_scores(result.retrieval, example),
            "citations": citation_scores(result, example),
            "support": factual_support_scores(annotations.get(result.sample_id)),
        }
        rows.append(row)
    statuses = Counter(result.response.status.value for result in results)
    totals = [
        sum(usage.total_tokens for usage in result.usage)
        if all(usage.total_tokens is not None for usage in result.usage) else None
        for result in results
    ]
    report = {
        "sample_count": len(results),
        "citation_schemes": sorted({r.integration.get("citation_scheme", "shared_corpus_citation_id")
                                    for r in results}),
        "gold_unit": "native_passage" if any(native) else "grounding_fragment_group",
        "reference_file": str(examples_path.resolve()),
        "is_development_mock": any(result.is_mock for result in results),
        "response_f1": _average(row["response"]["f1"] for row in rows),
        "response_em": _average(row["response"]["em"] for row in rows),
        "response_scored_count": sum(row["response"]["f1"] is not None for row in rows),
        "retrieval_stage": "post-fusion/post-rerank candidates before final context truncation",
        "retrieval": {
            f"@{k}": {
                metric: _average(row["retrieval"][f"@{k}"][metric] for row in rows)
                for metric in ["recall", "hit", "complete_coverage"]
            } for k in [1, 5, 10]
        },
        "retrieval_scored_count": sum(
            row["retrieval"]["@1"]["recall"] is not None for row in rows
        ),
        "citation_evidence_validity": _average(row["citations"]["evidence_validity"] for row in rows),
        "citation_gold_precision": _average(row["citations"]["gold_precision"] for row in rows),
        "hallucination_rate": _average(row["support"]["hallucination_rate"] for row in rows),
        "grounding_accuracy": _average(row["support"]["grounding_accuracy"] for row in rows),
        "dialogue_consistency": _average(row["support"]["dialogue_consistency"] for row in rows),
        "task_completion_rate": _average(row["support"]["task_completed"] for row in rows),
        "support_annotation_count": len(set(ids) & set(annotations)),
        "status_counts": dict(statuses),
        "status_rates": {status: count / len(results) for status, count in statuses.items()},
        "average_latency_ms": mean(result.elapsed_ms for result in results),
        "p95_latency_ms": float(np.percentile([r.elapsed_ms for r in results], 95)),
        "average_retrieval_rounds": mean(r.retrieval_round for r in results),
        "average_retry_count": mean(r.retry_count for r in results),
        "average_model_calls": mean(len(r.model_calls) or len(r.usage) for r in results),
        "average_total_tokens": _average(totals),
        "complete_usage_count": sum(value is not None for value in totals),
        "token_sources": sorted({usage.source for r in results for usage in r.usage}),
        "notes": [
            "Unanswered samples with references receive F1/EM zero; evidence metrics exclude unlabelled cases.",
            "Citation correctness does not prove factual support.",
            "Hallucination/consistency/completion remain null unless external annotations are supplied.",
            "Development mini results are wiring checks, not final benchmark results.",
        ],
    }
    (directory / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (directory / "evaluation.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
    )
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "status", "f1", "em", "recall@1", "recall@5", "recall@10"])
        for row in rows:
            writer.writerow([
                row["sample_id"], row["status"], row["response"]["f1"], row["response"]["em"],
                *[row["retrieval"][f"@{k}"]["recall"] for k in [1, 5, 10]],
            ])
    return report
