"""Offline required-point evaluation for the main-model A/B/C run."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path
import sys
from statistics import mean
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "vendor"))
sys.path.insert(0, str(ROOT / "main/src"))

from prepare_main import model_config
from customer_service_rag.models.llm import build_llm


ANNOTATIONS = HERE / "data/cross_domain_20_annotations.json"


def tokens(usage):
    return sum(u.get("total_tokens") or (u.get("input_tokens") or 0)
               + (u.get("output_tokens") or 0) for u in usage)


def average(values):
    valid = [value for value in values if value is not None]
    return mean(valid) if valid else None


def source_doc_key(item):
    doc_id = item["doc_id"]
    if "/" in doc_id:
        doc_id = unquote(doc_id.split("/", 1)[1])
    return item["domain"], doc_id


def evidence_metrics(evidence, sample, k):
    """Final-context metrics over annotated source documents and spans."""
    gold = {source_doc_key(item) for item in sample["evidence"]}
    selected = evidence[:k]
    predicted = {source_doc_key(item) for item in selected}
    overlap = gold & predicted
    precision = len(overlap) / len(predicted) if predicted else 0.0
    recall = len(overlap) / len(gold) if gold else None
    f1 = (2 * precision * recall / (precision + recall)
          if recall is not None and precision + recall else 0.0)
    gold_spans = {(source_doc_key(item), str(item["span_id"]))
                  for item in sample["evidence"] if item.get("span_id") is not None}
    seen_spans = {(source_doc_key(item), str(span))
                  for item in selected for span in item.get("span_ids", [])}
    return {"source_doc_precision": precision, "source_doc_recall": recall,
            "source_doc_f1": f1, "source_doc_hit": float(bool(overlap)),
            "source_doc_complete": float(gold <= predicted) if gold else None,
            "annotated_span_recall": (len(gold_spans & seen_spans) / len(gold_spans)
                                      if gold_spans else None)}


def citation_metrics(row, arm, sample):
    answer = row[arm]["generation"]["answer"]
    citations = set(answer["citations"])
    available = {item["passage_id"]: item for item in row[arm]["evidence"]}
    gold = {source_doc_key(item) for item in sample["evidence"]}
    if not citations:
        value = 0.0 if answer["status"] == "answered" else None
        return {"citation_evidence_validity": value,
                "cited_source_doc_precision": value}
    return {"citation_evidence_validity": len(citations & available.keys()) / len(citations),
            "cited_source_doc_precision": sum(
                source_doc_key(available[c]) in gold for c in citations if c in available
            ) / len(citations)}


async def grade_missing(config, rows, samples, output, graded):
    seen = {entry["sample_id"] for entry in graded}
    llm = build_llm(config.models.llm, root=config.project_root)
    with output.open("a", encoding="utf-8") as stream:
        for row in rows:
            sid = row["sample_id"]
            if sid in seen:
                continue
            sample = samples[sid]
            points = [{"point_id": p["point_id"], "expected_fact": p["expected_fact"],
                       "conditions": p["conditions"]} for p in sample["required_points"]]
            valid = {p["point_id"] for p in points}
            answers = {arm: row[arm]["generation"]["answer"] for arm in "ABC"}
            prompt = (
                "Independently score whether each answer explicitly covers each required point. "
                "Credit a point only if the substantive fact and relevant qualification are present. "
                "Do not credit retrieved evidence that is omitted from the answer. "
                "A status of error or insufficient_evidence covers no points. "
                'Return only JSON {"covered":{"A":[point_id],"B":[point_id],"C":[point_id]}}. '
                "Use only the supplied IDs.\n" + json.dumps({"question": row["question"],
                    "required_points": points, "answers": answers}, ensure_ascii=False)
            )
            before = len(llm.usage)
            completion = await llm.complete(prompt, task="evaluate", payload={})
            result = json.loads(completion.text)
            covered = result.get("covered")
            if not isinstance(covered, dict) or set(covered) != set("ABC"):
                raise ValueError(f"Malformed grade for {sid}")
            for arm in "ABC":
                if not isinstance(covered[arm], list) or set(covered[arm]) - valid:
                    raise ValueError(f"Invalid point IDs for {sid}/{arm}")
                if answers[arm]["status"] != "answered":
                    covered[arm] = []
            entry = {"sample_id": sid, "covered": covered,
                     "judge_usage": [u.model_dump(mode="json") for u in llm.usage[before:]]}
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stream.flush()
            graded.append(entry)
            seen.add(sid)
            print(f"{len(graded):02d}/{len(rows)} {sid}", flush=True)


def summarize(config, rows, samples, graded, run_dir):
    by_id = {x["sample_id"]: x for x in graded}
    summary = {"sample_count": len(rows), "judged_count": len(graded),
               "judge": config.models.llm.model,
               "warning": "Same-model judge and synthetic pending-review rubric; provisional answer scores.",
               "mean_shared_planner_tokens": mean(tokens(r["planner_usage"]) for r in rows),
               "offline_judge_total_tokens": sum(tokens(g["judge_usage"]) for g in graded),
               "arms": {}}
    def arm_usage(row, arm):
        calls = [row[arm]["generation"]]
        if arm == "C":
            calls += list(row[arm]["subanswers"].values())
        return [u for call in calls for u in call["usage"]]

    for arm in "ABC":
        fractions = [len(by_id[r["sample_id"]]["covered"][arm]) / len(samples[r["sample_id"]]["required_points"])
                     for r in rows]
        summary["arms"][arm] = {
            "required_point_coverage": mean(fractions),
            "all_points_covered_count": sum(x == 1 for x in fractions),
            "source_doc_recall": mean(r[arm]["retrieval_metrics"]["source_doc_recall"] for r in rows),
            "error_count": sum(r[arm]["generation"]["answer"]["status"] == "error" for r in rows),
            "mean_generation_tokens": mean(tokens(arm_usage(r, arm)) for r in rows),
            "mean_generation_calls": mean(len(arm_usage(r, arm)) for r in rows),
        }
        for k in (1, 3, 6):
            values = [evidence_metrics(r[arm]["evidence"], samples[r["sample_id"]], k)
                      for r in rows]
            summary["arms"][arm][f"final_context_at_{k}"] = {
                key: average(v[key] for v in values)
                for key in values[0]
            }
        cited = [citation_metrics(r, arm, samples[r["sample_id"]]) for r in rows]
        summary["arms"][arm]["citation_evidence_validity"] = average(
            v["citation_evidence_validity"] for v in cited)
        summary["arms"][arm]["cited_source_doc_precision"] = average(
            v["cited_source_doc_precision"] for v in cited)
        summary["arms"][arm]["status_counts"] = {
            status: sum(r[arm]["generation"]["answer"]["status"] == status for r in rows)
            for status in sorted({r[arm]["generation"]["answer"]["status"] for r in rows})
        }
        if all("pipeline_usage" in r[arm] for r in rows):
            summary["arms"][arm].update({
                "mean_pipeline_tokens": mean(tokens(r[arm]["pipeline_usage"]) for r in rows),
                "mean_pipeline_calls": mean(len(r[arm]["pipeline_usage"]) for r in rows),
                "mean_evaluations": mean(r[arm]["workflow"].get("evaluations", 0)
                    + (sum(s["workflow"].get("evaluations", 0)
                           for s in r[arm].get("subanswers", {}).values()) if arm == "C" else 0)
                    for r in rows),
                "mean_retries": mean(r[arm]["workflow"].get("retry_count", 0)
                    + (sum(s["workflow"].get("retry_count", 0)
                           for s in r[arm].get("subanswers", {}).values()) if arm == "C" else 0)
                    for r in rows),
                "mean_added_seconds": mean(r[arm]["added_elapsed_seconds"] for r in rows),
                "final_evaluation_pass_rate": mean(
                    r[arm]["workflow"]["stop_reason"] == "evaluation_passed" for r in rows),
                "initial_evaluation_pass_rate": mean(
                    r[arm]["workflow"].get("initial_feedback", {}).get("decision") == "pass"
                    for r in rows),
                "mean_optimization_actions": mean(
                    len(r[arm]["workflow"]["actions"])
                    + (sum(len(s["workflow"]["actions"])
                           for s in r[arm].get("subanswers", {}).values()) if arm == "C" else 0)
                    for r in rows),
                "stop_reason_counts": {
                    reason: sum(r[arm]["workflow"]["stop_reason"] == reason for r in rows)
                    for reason in sorted({r[arm]["workflow"]["stop_reason"] for r in rows})
                },
                "last_evaluated_candidate_scores": {
                    score: average(
                        r[arm]["workflow"]["feedback_history"][-1][score]
                        if r[arm]["workflow"]["feedback_history"] else None
                        for r in rows)
                    for score in ("relevance", "sufficiency", "groundedness",
                                  "dialogue_consistency")
                },
            })
            if arm == "C":
                summary["arms"][arm]["domain_initial_evaluation_pass_rate"] = mean(
                    s["workflow"].get("initial_feedback", {}).get("decision") == "pass"
                    for r in rows for s in r[arm]["subanswers"].values())
                summary["arms"][arm]["domain_evaluation_pass_rate"] = mean(
                    s["workflow"]["stop_reason"] == "evaluation_passed"
                    for r in rows for s in r[arm]["subanswers"].values())
                summary["arms"][arm]["domain_status_counts"] = {
                    status: sum(s["answer"]["status"] == status
                                for r in rows for s in r[arm]["subanswers"].values())
                    for status in sorted({s["answer"]["status"]
                        for r in rows for s in r[arm]["subanswers"].values()})
                }
    summary["routing_exact_match_rate"] = mean(
        set(r["predicted_domains"]) == set(r["gold_domains"]) for r in rows)
    summary["response_f1"] = None
    summary["response_em"] = None
    summary["response_reference_note"] = (
        "Main's token F1 and exact match require independent reference answers; "
        "this synthetic annotation set has required points but no reference answers. "
        "The reported source_doc_f1 is retrieval document-set F1, not response F1."
    )
    (run_dir / "main_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (run_dir / "case_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "predicted_domains", "gold_domains", "required_points",
                         "A_covered", "B_covered", "C_covered", "A_source_doc_recall",
                         "B_source_doc_recall", "C_source_doc_recall", "A_generation_tokens",
                         "B_generation_tokens", "C_generation_tokens",
                         "A_source_doc_f1@6", "B_source_doc_f1@6", "C_source_doc_f1@6",
                         "A_status", "B_status", "C_status",
                         "A_pipeline_tokens", "B_pipeline_tokens", "C_pipeline_tokens",
                         "A_evaluations", "B_evaluations", "C_evaluations",
                         "A_retries", "B_retries", "C_retries"])
        for row in rows:
            sid = row["sample_id"]
            writer.writerow([sid, "+".join(row["predicted_domains"]), "+".join(row["gold_domains"]),
                len(samples[sid]["required_points"]),
                *[len(by_id[sid]["covered"][arm]) for arm in "ABC"],
                *[row[arm]["retrieval_metrics"]["source_doc_recall"] for arm in "ABC"],
                *[tokens(arm_usage(row, arm)) for arm in "ABC"],
                *[evidence_metrics(row[arm]["evidence"], samples[sid], 6)["source_doc_f1"]
                  for arm in "ABC"],
                *[row[arm]["generation"]["answer"]["status"] for arm in "ABC"],
                *[tokens(row[arm]["pipeline_usage"]) if "pipeline_usage" in row[arm] else ""
                  for arm in "ABC"],
                *[(row[arm]["workflow"].get("evaluations", 0)
                   + (sum(s["workflow"].get("evaluations", 0)
                          for s in row[arm].get("subanswers", {}).values()) if arm == "C" else 0))
                  if "workflow" in row[arm] else "" for arm in "ABC"],
                *[(row[arm]["workflow"].get("retry_count", 0)
                   + (sum(s["workflow"].get("retry_count", 0)
                          for s in row[arm].get("subanswers", {}).values()) if arm == "C" else 0))
                  if "workflow" in row[arm] else "" for arm in "ABC"]])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = model_config()
    rows = [json.loads(line) for line in (args.run_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    samples = {item["sample_id"]: item for item in json.loads(ANNOTATIONS.read_text(encoding="utf-8"))}
    output = args.run_dir / "point_grades.jsonl"
    graded = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] if output.exists() else []
    if len({entry["sample_id"] for entry in graded}) < len(rows):
        asyncio.run(grade_missing(config, rows, samples, output, graded))
    summarize(config, rows, samples, graded, args.run_dir)
