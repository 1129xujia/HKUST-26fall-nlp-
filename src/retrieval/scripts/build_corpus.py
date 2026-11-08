"""Build section-level passages.jsonl from MultiDoc2Dial."""

import argparse
import json
from pathlib import Path

from retrieval.corpus import TITLE_SECTION_MODES, iter_passages, title_section_stats, write_passages_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--documents",
        type=Path,
        default=Path("multidoc2dial/multidoc2dial_doc.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/passages.jsonl"),
    )
    parser.add_argument(
        "--title-section-mode",
        choices=sorted(TITLE_SECTION_MODES),
        default="independent",
        help="Keep title passages, merge title text into body, or store it only as body metadata.",
    )
    parser.add_argument("--segmentation", choices=["section", "struct"], default="section")
    args = parser.parse_args()

    count = write_passages_jsonl(
        iter_passages(args.documents, title_section_mode=args.title_section_mode, segmentation=args.segmentation),
        args.output,
    )
    metadata = {
        "segmentation": args.segmentation,
        "title_section_mode": args.title_section_mode,
        "passage_count": count,
        **(title_section_stats(args.documents) if args.segmentation == "section" else {}),
    }
    metadata_path = args.output.parent / "corpus_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {count} passages to {args.output}")
    print(f"Wrote corpus metadata to {metadata_path}")


if __name__ == "__main__":
    main()
