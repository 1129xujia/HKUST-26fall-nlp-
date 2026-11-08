"""CLI for the independent Part 1/2/3 demonstration."""

import argparse
import json
from pathlib import Path

from retrieval.config import RetrievalConfig
from retrieval.pipeline import HybridRetrievalPipeline

from .orchestrator import EndToEndDemo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--mode", choices=["bm25", "dense", "rrf", "full"], default="rrf")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--history", type=Path, help="JSON list of role/content dialogue turns")
    args = parser.parse_args()
    config = RetrievalConfig.for_mode(args.mode, artifact_dir=args.artifact_dir)
    pipeline = HybridRetrievalPipeline.from_artifacts(
        artifact_dir=args.artifact_dir, config=config, device=args.device
    )
    history = json.loads(args.history.read_text(encoding="utf-8")) if args.history else None
    result = EndToEndDemo(pipeline).run(
        args.question, history=history, top_k=args.top_k
    ).to_dict()
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
