"""Reproducible acceptance run with persisted evidence and balanced samples."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from retrieval.config import RetrievalConfig
from retrieval.corpus import load_passages_jsonl, passage_fingerprint
from retrieval.evaluation import build_validation_samples, evaluate_pipeline
from retrieval.pipeline import HybridRetrievalPipeline
from retrieval.reranker import NoOpReranker
from retrieval.integration_demo.orchestrator import EndToEndDemo


def balanced_samples(samples, limit):
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.domain].append(sample)
    chosen = []
    for row in range(max(map(len, groups.values()), default=0)):
        for domain in sorted(groups):
            if row < len(groups[domain]):
                chosen.append(groups[domain][row])
                if len(chosen) == limit:
                    return chosen
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Require both real models and all four ablations")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/acceptance"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.limit < 4:
        parser.error("--limit must be at least 4 to cover every domain")
    root = Path(__file__).resolve().parents[3]
    os.chdir(root)
    os.environ.setdefault("HF_HOME", str(root / ".model_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "status": "running", "full": args.full, "requested_samples": args.limit,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
           "PYTHONPATH": str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    unit = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "src/retrieval/tests", "src/retrieval/integration_demo/tests"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
    )
    (args.output_dir / "unit_tests.txt").write_text(unit.stdout, encoding="utf-8")
    print(unit.stdout, flush=True)
    unit.check_returncode()
    passages = load_passages_jsonl(args.artifact_dir / "passages.jsonl")
    ids = {p.passage_id for p in passages}
    assert len(ids) == len(passages), "Passage IDs must be unique"
    assert {p.domain for p in passages} == {"dmv", "ssa", "va", "studentaid"}
    assert all(p.text.strip() and p.span_ids for p in passages)
    samples = balanced_samples(build_validation_samples(
        document_path=root / "multidoc2dial/multidoc2dial_doc.json",
        dialogue_path=root / "multidoc2dial/multidoc2dial_dial_validation.json",
    ), args.limit)
    missing_gold = sorted({pid for s in samples for pid in s.gold_passage_ids} - ids)
    assert not missing_gold, f"Gold references missing from corpus: {missing_gold[:3]}"
    config = RetrievalConfig.for_mode(
        "full" if args.full else "bm25",
        artifact_dir=args.artifact_dir,
        allow_reranker_fallback=False,
    )
    print("Loading real retrieval artifacts...", flush=True)
    complete = HybridRetrievalPipeline.from_artifacts(
        artifact_dir=args.artifact_dir, config=config, device=args.device,
    )
    modes = ["bm25", "dense", "rrf", "full"] if args.full else ["bm25"]
    summaries = {}
    for mode in modes:
        print(f"Evaluating {mode}: {len(samples)} balanced validation samples...", flush=True)
        mode_config = RetrievalConfig.for_mode(
            mode, artifact_dir=args.artifact_dir, allow_reranker_fallback=False,
        )
        pipeline = HybridRetrievalPipeline(
            dense_retriever=complete.dense_retriever,
            bm25_retriever=complete.bm25_retriever,
            config=mode_config,
            reranker=complete.reranker if mode == "full" else NoOpReranker(),
            deduplicator=complete.deduplicator,
        )
        measured = evaluate_pipeline(
            pipeline, samples,
            on_progress=lambda done, total: print(
                f"  {mode}: {done}/{total}", flush=True
            ) if done % 10 == 0 or done == total else None,
        )
        measured["configuration"] = {
            "mode": mode, "query_mode": "history", "history_turns": 4,
            "domain_filter": True, "sample_strategy": "deterministic_round_robin_by_domain",
            "sample_ids": [s.sample_id for s in samples],
        }
        (args.output_dir / f"evaluation_{mode}.json").write_text(json.dumps(measured, indent=2, allow_nan=False), encoding="utf-8")
        summaries[mode] = measured["summary"]
        print(json.dumps(measured["summary"]), flush=True)
    demos = []
    for question in [
        "How do I renew a driver license?",
        "How can I apply for Social Security retirement benefits?",
        "How do veterans apply for VA health care?",
        "How do I apply for federal student aid with FAFSA?",
    ]:
        result = EndToEndDemo(complete).run(question).to_dict()
        assert result["quality"]["grounded"] and result["quality"]["citations_valid"]
        demos.append({"question": question, **result})
    (args.output_dir / "demo_cases.json").write_text(json.dumps(demos, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    summary = {
        "status": "passed", "full": args.full,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "corpus_passages": len(passages), "corpus_domains": dict(Counter(p.domain for p in passages)),
        "corpus_fingerprint": passage_fingerprint(passages),
        "validation_domains": dict(Counter(s.domain for s in samples)),
        "models": {"embedding": config.embedding_model, "reranker": config.reranker_model},
        "runtime_notes": complete.runtime_notes, "metrics": summaries,
        "demo_cases_passed": len(demos),
        "limits": "Small balanced validation smoke test; citation validity is structural, not semantic entailment.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Acceptance passed. See {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
