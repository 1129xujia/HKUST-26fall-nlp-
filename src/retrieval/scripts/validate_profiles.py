"""Local real-model acceptance; four full-corpus indexes and mock downstream LLM."""

import argparse
import asyncio
import json
from pathlib import Path
import time
import tempfile
import hashlib

from retrieval.profiles import profile_config
from retrieval.corpus import load_passages_jsonl, passage_fingerprint, iter_passages
from retrieval.pipeline import HybridRetrievalPipeline
from retrieval.schemas import RetrievalRequest
from retrieval.service import build_profile
from retrieval.native_main import run_native
from customer_service_rag.models.llm import MockLLM
from customer_service_rag.schemas import DialogueInput


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, default=Path("artifacts/indexes"))
    parser.add_argument("--output", type=Path, default=Path("reports/retrieval_profiles/local_validation.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-docs-per-domain", type=int, default=0,
                        help="0 uses the full corpus; positive values run a smaller real-data integration check")
    args = parser.parse_args()
    if args.max_docs_per_domain < 0:
        parser.error("max-docs-per-domain must be nonnegative")
    original = args.documents
    temporary = tempfile.TemporaryDirectory(prefix="retrieval-profile-validation-")
    if args.max_docs_per_domain:
        payload = json.loads(original.read_text())
        args.documents = Path(temporary.name) / "documents.json"
        args.documents.write_text(json.dumps({"doc_data": {
            domain: dict(list(documents.items())[:args.max_docs_per_domain])
            for domain, documents in payload["doc_data"].items()
        }}, ensure_ascii=False))
    report = {"scope": "Real encoders/FAISS/BM25/Cross-Encoder; mock generation/evaluation, not a quality benchmark",
              "device": args.device, "max_docs_per_domain": args.max_docs_per_domain,
              "original_documents_sha256": hashlib.sha256(original.read_bytes()).hexdigest(), "profiles": []}
    query = "How do I replace a lost driver's license?"
    for segmentation in ("section", "struct"):
        for chain in ("rerank", "finetuned"):
            config = profile_config(segmentation=segmentation, chain=chain, index_root=args.index_root)
            print(f"Validating {segmentation}/{chain}", flush=True)
            started = time.perf_counter()
            built = not config.artifact_dir.exists()
            if built:
                build_profile(config, args.documents, device=args.device)
            pipeline = HybridRetrievalPipeline.from_artifacts(artifact_dir=config.artifact_dir,
                                                               config=config, device=args.device)
            evidence = pipeline.retrieve(RetrievalRequest("User: " + query, top_k=10))
            passages = load_passages_jsonl(config.artifact_dir / "passages.jsonl")
            assert passage_fingerprint(passages) == passage_fingerprint(iter_passages(
                args.documents, segmentation=segmentation)), "Existing acceptance index has a different corpus; use a fresh --index-root"
            assert len(evidence) == 10 and len({e.passage_id for e in evidence}) == 10
            assert all(bool(e.text.strip()) for e in evidence)
            assert all(("::structure_" in e.passage_id) == (segmentation == "struct") for e in evidence)
            if chain == "finetuned":
                assert all(e.source_retrievers == ["dense"] and e.bm25_rank is None and e.rrf_score == 0 for e in evidence)
            result = asyncio.run(run_native(pipeline, DialogueInput(conversation_id="profile-acceptance",
                current_message=query), MockLLM(), corpus_version=passage_fingerprint(passages), top_k=3))
            assert result["stop_reason"] == "evaluation_passed", result
            assert result["response"]["citations"]
            assert set(result["response"]["citations"]) <= {e["passage_id"] for e in result["initial_evidence"]}
            row = {"segmentation": segmentation, "chain": chain,
                   "passage_count": len(passages), "built_index_this_run": built,
                   "artifact_dir": str(config.artifact_dir),
                   "embedding_model": config.embedding_model,
                   "local_model_fingerprint": json.loads((config.artifact_dir / "dense_metadata.json").read_text())["local_model_fingerprint"],
                   "elapsed_s_including_build_and_load": time.perf_counter()-started,
                   "top10": [e.to_dict() for e in evidence], "native_main": result}
            report["profiles"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
            print(f"Passed {segmentation}/{chain}: {len(passages)} passages", flush=True)
            del pipeline
    print(f"All four profiles passed; report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
