"""Project index adapters around the team's retrieval algorithms.

Shared NumPy/MiniLM and native FAISS/BGE indexes both execute the same BM25,
fusion, reranking and optimizer adapter. Only asset loading and ID conversion
differ; planning, generation and loop control live in the main orchestrator.
"""

from __future__ import annotations

from dataclasses import replace
from importlib.util import find_spec
import json
from pathlib import Path
from threading import Lock

import numpy as np

from customer_service_rag.data.indexing import build_index, corpus_digest, load_index
from customer_service_rag.data.io import load_passages
from customer_service_rag.models.embeddings import build_embedding
from customer_service_rag.models.runtime import load_runtime
from customer_service_rag.schemas import (
    Passage as ProjectPassage,
    RetrievalCandidate, RetrievalHit, RetrievalOptions, RetrievalResult,
)
from customer_service_rag.optimization.retrieval_adapter import Part2RetrievalAdapter
from customer_service_rag.optimization.schemas import ExecutionOptions
from customer_service_rag.retrieval.bm25 import BM25Retriever
from customer_service_rag.retrieval.config import RetrievalConfig
from customer_service_rag.retrieval.corpus import load_passages_jsonl, passage_fingerprint
from customer_service_rag.retrieval.pipeline import HybridRetrievalPipeline
from customer_service_rag.retrieval.profiles import profile_config
from customer_service_rag.retrieval.reranker import CrossEncoderReranker
from customer_service_rag.retrieval.schemas import Passage, SearchHit


class SharedDenseRetriever:
    """Expose a checksum-validated project index through the team's protocol."""

    def __init__(self, passages, index, embedding):
        self.passages, self.index, self.embedding = passages, index, embedding
        self.lock = Lock()

    def search(self, query, *, domain=None, top_k=20):
        with self.lock:
            vector = self.embedding.encode([query])[0]
        scores = self.index.vectors @ vector
        rows = [int(row) for row in np.argsort(-scores, kind="stable")
                if (domain is None or self.passages[int(row)].domain == domain)
                and (getattr(self.embedding, "backend", "hash") != "hash" or scores[row] > 0)]
        return [SearchHit(self.passages[row], "dense", rank, float(scores[row]), query)
                for rank, row in enumerate(rows[:top_k], 1)]


class LocalPairPredictor:
    """Run the team's CrossEncoder with the project's verified local weights."""

    def __init__(self, config, root):
        self.config = config
        self.runtime = load_runtime(
            str((root / config.local_path).resolve()), "reranker",
            config.model, config.revision, config.num_threads,
        )

    def predict(self, pairs, *, batch_size=16, show_progress_bar=False):
        runtime, scores = self.runtime, []
        with runtime.lock, runtime.torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start:start + batch_size]
                inputs = runtime.tokenizer(
                    [pair[0] for pair in batch], [pair[1] for pair in batch],
                    padding=True, truncation="only_second", max_length=self.config.max_length,
                    return_tensors="pt",
                )
                logits = runtime.model(**inputs).logits
                if logits.shape[-1] != 1:
                    raise ValueError("The configured reranker must return one relevance logit")
                scores.extend(logits[:, 0].cpu().tolist())
        return np.asarray(scores)


def result_options(options: ExecutionOptions) -> RetrievalOptions:
    """Summarize the executed options in the shared saved-result contract."""
    branches = [name for name in ("dense", "bm25")
                if getattr(options, f"{name}_top_k") > 0
                and getattr(options, f"{name}_weight") > 0]
    return RetrievalOptions(
        branches=branches,
        per_branch_top_k=max(options.dense_top_k, options.bm25_top_k), fusion="rrf",
        rrf_k=options.rrf_k, candidate_top_k=options.candidate_top_k,
        final_top_k=options.final_top_k, rerank=options.rerank,
    )


def finetuned_query(dialogue):
    """Match fine-tuning's last-four-visible-turn representation."""
    turns = [(m.role, m.text) for m in dialogue.dialogue_history if m.role in {"user", "agent"}]
    turns.append(("user", dialogue.current_message))
    return "\n".join(f"{role.title()}: {' '.join(text.split())}" for role, text in turns[-4:])


def native_config_for(config):
    settings = config.retrieval
    native = profile_config(
        segmentation=settings.segmentation, chain=settings.chain,
        artifact_dir=config.resolve_path(config.paths.index_dir),
        embedding_model=settings.native_embedding_model,
        reranker_model=settings.native_reranker_model,
        project_root=config.project_root,
    )
    if settings.chain == "finetuned":
        if settings.branches != ["dense"] or settings.rerank:
            raise ValueError("The fine-tuned chain requires branches: [dense] and rerank: false")
    return replace(
        native,
        use_dense="dense" in settings.branches,
        use_bm25="bm25" in settings.branches,
        use_reranker=settings.rerank,
        dense_top_k=settings.dense_top_k,
        bm25_top_k=settings.bm25_top_k,
        fused_top_k=settings.candidate_top_k,
        final_top_k=settings.final_top_k,
        rrf_k=settings.rrf_k,
    )


def documents_for(config):
    explicit = config.retrieval.documents
    if explicit is not None:
        path = config.resolve_path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Document source does not exist: {path}")
        return path
    candidates = sorted(
        config.resolve_path(config.data.raw_dir).rglob("multidoc2dial_doc.json"),
        key=lambda path: (len(path.parts), path.as_posix()),
    )
    if not candidates:
        raise FileNotFoundError("Native indexing needs retrieval.documents or downloaded MultiDoc2Dial data")
    return candidates[0]


def build_index_for_config(config):
    """Build the selected backend without changing another backend's assets."""
    directory = config.resolve_path(config.paths.index_dir)
    if config.retrieval.backend == "shared":
        passages = load_passages(config.resolve_path(config.data.passages_path))
        embedding = build_embedding(config.models.embedding, config.project_root)
        return build_index(passages, embedding, directory)
    from customer_service_rag.retrieval.service import build_profile

    native = native_config_for(config)
    build_profile(native, documents_for(config), device=config.retrieval.device)
    return {
        "index_backend": "faiss", "artifact_dir": str(directory),
        **json.loads((directory / "service_manifest.json").read_text(encoding="utf-8")),
        "dense": json.loads((directory / "dense_metadata.json").read_text(encoding="utf-8"))
        if (directory / "dense_metadata.json").is_file() else None,
    }


def describe_assets(config):
    """Describe required files without loading models or downloading anything."""
    directory = config.resolve_path(config.paths.index_dir)
    native = config.retrieval.backend == "native"
    files = ["passages.jsonl", "corpus_metadata.json", "service_manifest.json"] if native else [
        "passages.jsonl", "manifest.json", "vectors.npy",
    ]
    if native and "dense" in config.retrieval.branches:
        files.extend(["dense_metadata.json", "faiss.index"])
    if native and "bm25" in config.retrieval.branches:
        files.extend(["bm25.pkl", "bm25_metadata.json"])
    return {
        "backend": "faiss" if native else "shared_numpy", "index_dir": str(directory),
        "files": {name: (directory / name).is_file() for name in files},
        "optional_dependencies": {name: find_spec(name) is not None for name in (
            "sentence_transformers", "faiss",
        )} if native else {},
        "segmentation": config.retrieval.segmentation if native else "character_window",
        "chain": config.retrieval.chain if native else "hybrid",
    }


class ProjectRetriever:
    """Load one corpus once and expose one optimizer-compatible retrieval API."""

    def __init__(self, config):
        self.config = config
        if config.retrieval.backend == "native":
            self._load_native()
        else:
            self._load_shared()
        self.adapter = Part2RetrievalAdapter(self.pipeline, corpus_version=self.corpus_version)
        self.reranker = self.pipeline.reranker

    @classmethod
    def from_pipeline(cls, config, pipeline, *, corpus_version):
        """Inject already-loaded team components into the same application API."""
        self = cls.__new__(cls)
        self.config, self.pipeline, self.corpus_version = config, pipeline, corpus_version
        self.corpus, self.passages = {}, {}
        for retriever in (pipeline.dense_retriever, pipeline.bm25_retriever):
            for passage in getattr(retriever, "passages", []):
                self._register_passage(passage)
        native = config.retrieval.backend == "native"
        self.index_backend = "faiss" if native else "shared_numpy"
        self.embedding_backend = "sentence_transformers" if native else config.models.embedding.backend
        self.is_mock_embedding = not native and self.embedding_backend == "hash"
        self.citation_scheme = f"native_doc::{config.retrieval.segmentation}" if native else "shared_corpus_citation_id"
        self.index_manifest = {"injected_pipeline": True, "passage_count": len(self.passages)}
        self.adapter = Part2RetrievalAdapter(pipeline, corpus_version=corpus_version)
        self.reranker = pipeline.reranker
        return self

    def _register_passage(self, passage):
        key = passage.passage_id
        if key in self.passages:
            return
        self.corpus[key] = passage
        native = self.config.retrieval.backend == "native"
        passage_id = key if native else key.removeprefix(f"{passage.doc_id}:")
        self.passages[key] = ProjectPassage(
            doc_id=passage.doc_id, passage_id=passage_id, title=passage.title,
            text=passage.text, domain=passage.domain, span_ids=passage.span_ids,
            char_start=getattr(passage, "start", None), char_end=getattr(passage, "end", None),
            source=f"native_{self.config.retrieval.segmentation}" if native else "synthetic",
            original_doc_id=passage.doc_id,
        )

    def _load_shared(self):
        config = self.config
        self.embedding = build_embedding(config.models.embedding, config.project_root)
        index = load_index(config.resolve_path(config.paths.index_dir), self.embedding)
        expected = load_passages(config.resolve_path(config.data.passages_path))
        if corpus_digest(expected) != index.manifest["corpus_sha256"]:
            raise ValueError("Configured corpus differs from index; run cs-rag index again")
        self.passages = {p.citation_id: p for p in index.passages}
        converted = [Passage(
            passage_id=p.citation_id, doc_id=p.doc_id, domain=p.domain, title=p.title,
            section_title="", parent_titles=[], text=p.text, span_ids=p.span_ids,
            start=p.char_start, end=p.char_end,
        ) for p in index.passages]
        self.corpus = {p.passage_id: p for p in converted}
        branches = config.retrieval.branches
        dense = SharedDenseRetriever(converted, index, self.embedding) if "dense" in branches else None
        bm25 = BM25Retriever.build(converted) if "bm25" in branches else None
        reranker = None
        if config.retrieval.rerank:
            settings = config.models.reranker
            reranker = CrossEncoderReranker(
                f"{settings.model}@{settings.revision}",
                model=LocalPairPredictor(settings, config.project_root),
                batch_size=settings.batch_size,
            )
        self.pipeline = HybridRetrievalPipeline(
            dense_retriever=dense, bm25_retriever=bm25, reranker=reranker,
            config=RetrievalConfig(
                embedding_model=self.embedding.model_name,
                use_dense=dense is not None, use_bm25=bm25 is not None,
                use_reranker=reranker is not None, use_deduplication=False,
                dense_top_k=config.retrieval.dense_top_k,
                bm25_top_k=config.retrieval.bm25_top_k,
                fused_top_k=config.retrieval.candidate_top_k,
                final_top_k=config.retrieval.final_top_k,
                rrf_k=config.retrieval.rrf_k,
            ),
        )
        self.index_backend = "shared_numpy"
        self.embedding_backend = config.models.embedding.backend
        self.is_mock_embedding = self.embedding_backend == "hash"
        self.citation_scheme = "shared_corpus_citation_id"
        self.index_manifest = index.manifest
        self.corpus_version = index.manifest["corpus_sha256"]

    def _load_native(self):
        config = self.config
        native = native_config_for(config)
        self.pipeline = HybridRetrievalPipeline.from_artifacts(
            artifact_dir=native.artifact_dir, config=native, device=config.retrieval.device,
        )
        corpus = load_passages_jsonl(native.artifact_dir / "passages.jsonl")
        self.corpus = {p.passage_id: p for p in corpus}
        if len(self.corpus) != len(corpus):
            raise ValueError("Native corpus contains duplicate passage IDs")
        self.passages = {p.passage_id: ProjectPassage(
            doc_id=p.doc_id, passage_id=p.passage_id, title=p.title, text=p.text,
            domain=p.domain, span_ids=p.span_ids, char_start=p.start, char_end=p.end,
            source=f"native_{native.segmentation}", original_doc_id=p.doc_id,
        ) for p in corpus}
        self.index_backend = "faiss"
        self.embedding_backend = "sentence_transformers"
        self.is_mock_embedding = False
        self.citation_scheme = f"native_doc::{native.segmentation}"
        self.index_manifest = {}
        for name in ("corpus_metadata", "dense_metadata", "service_manifest"):
            path = native.artifact_dir / f"{name}.json"
            if path.is_file():
                self.index_manifest[name] = json.loads(path.read_text(encoding="utf-8"))
        self.corpus_version = passage_fingerprint(corpus)

    def options(self, options: RetrievalOptions) -> ExecutionOptions:
        native = self.config.retrieval.backend == "native"
        if native and self.config.retrieval.chain == "finetuned" and (
            options.branches != ["dense"] or options.rerank
        ):
            raise ValueError("Fine-tuned retrieval cannot enable BM25 or reranking")
        return ExecutionOptions(
            dense_top_k=self.config.retrieval.dense_top_k if "dense" in options.branches else 0,
            bm25_top_k=self.config.retrieval.bm25_top_k if "bm25" in options.branches else 0,
            candidate_top_k=options.candidate_top_k, final_top_k=options.final_top_k,
            rrf_k=options.rrf_k, rerank=options.rerank,
            reranker_query_strategy=(self.pipeline.config.reranker_query_strategy if native else "identity"),
            reranker_use_sub_queries=False,
            reranker_skip_acknowledgements=self.pipeline.config.reranker_skip_acknowledgements if native else False,
            domain_filter=self.config.retrieval.domain_filter,
        )

    def plan_for_retrieval(self, plan, dialogue):
        if self.config.retrieval.backend == "native" and self.config.retrieval.chain == "finetuned":
            # The planner already applied use_history to the visible turns.
            return plan.model_copy(update={"rewritten_query": finetuned_query(plan), "sub_queries": []})
        return plan

    def result(self, snapshot) -> RetrievalResult:
        # An injected retriever need not own a full corpus; hydrate only its
        # returned passages through the same conversion used by saved indexes.
        for ranking in snapshot.rankings:
            for hit in ranking:
                self._register_passage(hit.passage)
        for candidate in snapshot.candidates:
            self._register_passage(candidate.passage)
        for evidence in snapshot.evidence:
            self._register_passage(evidence)
        by_passage = {}
        hits = []
        for ranking in snapshot.rankings:
            for hit in ranking:
                converted = RetrievalHit(
                    passage=self.passages[hit.passage.passage_id], source=hit.retriever,
                    query_id=hit.matched_query, score=hit.raw_score, rank=hit.rank,
                )
                hits.append(converted)
                by_passage.setdefault(hit.passage.passage_id, []).append(converted)
        return RetrievalResult(
            passages=[self.passages[e.passage_id] for e in snapshot.evidence],
            candidates=[RetrievalCandidate(
                passage=self.passages[c.passage.passage_id],
                hits=by_passage.get(c.passage.passage_id, []), fusion_score=c.rrf_score,
                rerank_score=c.rerank_score,
            ) for c in snapshot.candidates], hits=hits,
            elapsed_ms=snapshot.elapsed_ms, embedding_backend=self.embedding_backend,
            rerank_elapsed_ms=snapshot.rerank_elapsed_ms, rerank_model=snapshot.rerank_model,
        )

    def metadata_for_snapshot(self, snapshot):
        for evidence in snapshot.evidence:
            self._register_passage(evidence)
        native = self.config.retrieval.backend == "native"
        return {
            "segmentation": self.config.retrieval.segmentation if native else "character_window",
            "retrieval_chain": self.config.retrieval.chain if native else (
                "rerank" if snapshot.rerank_model is not None else "rrf"
            ),
            "evidence_metadata": {e.passage_id: self.corpus[e.passage_id].to_dict()
                                  for e in snapshot.evidence},
        }

    async def retrieve(self, query, options):
        return self.result(await self.adapter.retrieve(query, self.options(options)))
