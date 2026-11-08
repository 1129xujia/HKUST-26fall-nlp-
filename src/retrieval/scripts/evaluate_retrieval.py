"""Evaluate one retrieval configuration on MultiDoc2Dial validation turns."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from retrieval.config import RetrievalConfig
from retrieval.corpus import build_title_section_map
from retrieval.evaluation import build_validation_samples, evaluate_pipeline
from retrieval.pipeline import HybridRetrievalPipeline


def balanced_samples(samples, limit):
    """Round-robin samples across domains for small diagnostic evaluations."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=Path, default=Path("multidoc2dial/multidoc2dial_doc.json"))
    parser.add_argument("--dialogues", type=Path, default=Path("multidoc2dial/multidoc2dial_dial_validation.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    parser.add_argument("--mode", "--method", dest="mode", choices=["bm25", "dense", "rrf", "full"], default="rrf")
    parser.add_argument(
        "--query-mode",
        choices=["current", "history", "user_history", "current_first_user_history"],
        default="history",
    )
    parser.add_argument("--history-turns", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--balanced-by-domain", action="store_true")
    parser.add_argument("--no-domain-filter", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--title-section-diagnostics",
        action="store_true",
        help=(
            "Keep official metrics and additionally report diagnostics after mapping "
            "reliable title sections to their following body sections."
        ),
    )
    args = parser.parse_args()

    config = RetrievalConfig.for_mode(args.mode, artifact_dir=args.artifact_dir)

    samples = build_validation_samples(
        document_path=args.documents,
        dialogue_path=args.dialogues,
        query_mode=args.query_mode,
        history_turns=args.history_turns,
    )
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        selection_size = args.limit + args.offset
        if args.balanced_by_domain:
            samples = balanced_samples(samples, selection_size)
        samples = samples[args.offset:selection_size]
    elif args.offset:
        samples = samples[args.offset:]
    pipeline = HybridRetrievalPipeline.from_artifacts(
        artifact_dir=args.artifact_dir, config=config, device=args.device
    )
    result = evaluate_pipeline(
        pipeline,
        samples,
        use_domain_filter=not args.no_domain_filter,
        title_section_map=(
            build_title_section_map(args.documents)
            if args.title_section_diagnostics else None
        ),
    )
    result["configuration"] = {
        "mode": args.mode,
        "query_mode": args.query_mode,
        "domain_filter": not args.no_domain_filter,
        "balanced_by_domain": args.balanced_by_domain,
        "sample_offset": args.offset,
        "metric_contract": (
            "official_and_merged_title_diagnostic"
            if args.title_section_diagnostics else "official"
        ),
        "runtime_notes": pipeline.runtime_notes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote detailed results to {args.output}")


if __name__ == "__main__":
    main()
