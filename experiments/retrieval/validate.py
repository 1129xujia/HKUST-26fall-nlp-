"""Validate native assets through the single project orchestrator."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import time

from customer_service_rag.config import load_config
from customer_service_rag.models.llm import MockLLM
from customer_service_rag.orchestration.orchestrator import RAGOrchestrator
from customer_service_rag.orchestration.retrieval import ProjectRetriever, build_index_for_config
from customer_service_rag.schemas import DialogueInput
from customer_service_rag.retrieval.corpus import iter_passages, passage_fingerprint
from customer_service_rag.retrieval.schemas import RetrievalRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, default=Path("artifacts/indexes/acceptance"))
    parser.add_argument("--output", type=Path, default=Path("results/retrieval_profiles.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-docs-per-domain", type=int, default=1)
    args = parser.parse_args()
    if args.max_docs_per_domain < 0:
        parser.error("max-docs-per-domain must be nonnegative; 0 selects the full corpus")
    base = load_config(args.config)
    original = base.resolve_path(args.documents)
    output = base.resolve_path(args.output)
    report = {
        "scope": "Real native retrieval with mock generation/evaluation; not a quality benchmark",
        "device": args.device, "max_docs_per_domain": args.max_docs_per_domain,
        "original_documents_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "profiles": [],
    }
    query = "How do I replace a lost driver's license?"
    with tempfile.TemporaryDirectory(prefix="retrieval-profile-validation-") as temporary:
        documents = original
        if args.max_docs_per_domain:
            payload = json.loads(original.read_text(encoding="utf-8"))
            documents = Path(temporary) / "documents.json"
            documents.write_text(json.dumps({"doc_data": {
                domain: dict(list(values.items())[:args.max_docs_per_domain])
                for domain, values in payload["doc_data"].items()
            }}, ensure_ascii=False), encoding="utf-8")
        for segmentation in ("section", "struct"):
            for chain in ("rerank", "finetuned"):
                directory = base.resolve_path(args.index_root) / segmentation / chain
                config = load_config(args.config, overrides={
                    "models": {"llm": {"backend": "mock", "model": "mock-extractive-v1"}},
                    "planning": {"decompose": False},
                    "quality": {"enabled": True},
                    "optimization": {"enabled": True},
                    "budget": {"max_latency_seconds": 180, "max_total_tokens": 30000},
                    "retrieval": {
                        "backend": "native", "segmentation": segmentation, "chain": chain,
                        "device": args.device, "documents": str(documents),
                        "branches": ["dense"] if chain == "finetuned" else ["dense", "bm25"],
                        "rerank": chain == "rerank", "dense_top_k": 100, "bm25_top_k": 50,
                        "candidate_top_k": 50, "final_top_k": 3, "domain_filter": False,
                    },
                    "paths": {"index_dir": str(directory)},
                })
                print(f"Validating {segmentation}/{chain}", flush=True)
                started = time.perf_counter()
                built = not directory.exists()
                if built:
                    build_index_for_config(config)
                retriever = ProjectRetriever(config)
                evidence = retriever.pipeline.retrieve(RetrievalRequest("User: " + query, top_k=10))
                corpus = list(retriever.corpus.values())
                expected = iter_passages(documents, segmentation=segmentation)
                if passage_fingerprint(corpus) != passage_fingerprint(expected):
                    raise ValueError("Existing acceptance index has a different corpus; use a fresh --index-root")
                if len(evidence) != 10 or len({e.passage_id for e in evidence}) != 10:
                    raise ValueError("Native retrieval must return ten distinct passages")
                if any(not e.text.strip() for e in evidence):
                    raise ValueError("Native retrieval returned empty evidence")
                if any(("::structure_" in e.passage_id) != (segmentation == "struct") for e in evidence):
                    raise ValueError("Native evidence IDs do not match the selected segmentation")
                if chain == "finetuned" and any(
                    e.source_retrievers != ["dense"] or e.bm25_rank is not None or e.rrf_score != 0
                    for e in evidence
                ):
                    raise ValueError("Fine-tuned retrieval must preserve the Dense-only chain")
                runner = RAGOrchestrator(config, retriever=retriever, llm_factory=lambda: MockLLM())
                result = asyncio.run(runner.run(DialogueInput(
                    conversation_id="profile-acceptance", current_message=query,
                )))
                if result.stop_reason != "evaluation_passed" or not result.response.citations:
                    raise ValueError(f"Mock downstream integration failed: {result.stop_reason}")
                if not set(result.response.citations) <= {p.citation_id for p in result.retrieval.passages}:
                    raise ValueError("Generated citations are outside the final retrieval context")
                report["profiles"].append({
                    "segmentation": segmentation, "chain": chain, "passage_count": len(corpus),
                    "built_index_this_run": built, "artifact_dir": str(directory),
                    "index_manifest": retriever.index_manifest,
                    "elapsed_s_including_build_and_load": time.perf_counter() - started,
                    "top10": [e.to_dict() for e in evidence],
                    "application": result.model_dump(mode="json"),
                })
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(f"Passed {segmentation}/{chain}: {len(corpus)} passages", flush=True)
    print(f"All four profiles passed; report: {output}", flush=True)


if __name__ == "__main__":
    main()
