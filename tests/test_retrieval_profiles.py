"""Regression checks for selectable corpus/model profiles and service integration."""

import asyncio
import json

import pytest

from retrieval.config import RetrievalConfig
from retrieval.corpus import iter_passages
from retrieval.model_identity import local_model_fingerprint
from retrieval.pipeline import HybridRetrievalPipeline
from retrieval.profiles import profile_config
from retrieval.schemas import Passage, SearchHit, RetrievalRequest
from retrieval.exceptions import RetrievalModelMismatch
from optimization_v2.retrieval_adapter import Part2RetrievalAdapter
from customer_service_rag.schemas import DialogueInput, QueryPlan
from customer_service_rag.models.llm import MockLLM
from retrieval.native_main import run_native, finetuned_query


def test_struct_merges_adjacent_titles_but_not_distant_same_titles(tmp_path):
    spans = {}
    for i, (section, title) in enumerate([("a", "Shared"), ("b", "Shared"), ("c", "Other"), ("d", "Shared")]):
        spans[str(i)] = dict(id_sp=str(i), id_sec=section, title=title,
                             text_sp=f"body{i}", text_sec=f"body{i}", start_sp=i*10,
                             end_sp=i*10+5, parent_titles=[])
    source = tmp_path / "docs.json"
    source.write_text(json.dumps({"doc_data": {"dmv": {"doc": {"title": "Doc", "spans": spans}}}}))
    section = list(iter_passages(source, segmentation="section"))
    struct = list(iter_passages(source, segmentation="struct"))
    assert len(section) == 4 and len(struct) == 3
    assert struct[0].span_ids == ["0", "1"]
    assert struct[-1].span_ids == ["3"]
    assert {s for p in struct for s in p.span_ids} == set(spans)


@pytest.mark.parametrize("segmentation", ["section", "struct"])
@pytest.mark.parametrize("chain", ["rerank", "finetuned"])
def test_profiles_isolate_indexes_and_select_only_requested_models(tmp_path, segmentation, chain):
    config = profile_config(segmentation=segmentation, chain=chain, index_root=tmp_path,
                            embedding_model=str(tmp_path))
    assert config.artifact_dir == tmp_path / segmentation / chain
    assert config.use_reranker == (chain == "rerank")
    assert config.use_bm25 == (chain == "rerank")
    assert config.use_sub_queries == (chain == "rerank")


def test_model_identity_detects_weight_and_tokenizer_changes(tmp_path):
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"weights-one")
    initial = local_model_fingerprint(str(tmp_path))
    weight.write_bytes(b"weights-two")
    changed = local_model_fingerprint(str(tmp_path))
    assert initial != changed
    (tmp_path / "tokenizer.json").write_text('{"changed": true}')
    assert changed != local_model_fingerprint(str(tmp_path))
    weight.write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="LFS pointer"):
        local_model_fingerprint(str(tmp_path))


class Dense:
    def __init__(self):
        self.queries = []

    def search(self, query, *, domain=None, top_k=20):
        self.queries.append(query)
        # Deliberate tied cosine scores and reverse lexical IDs.
        return [SearchHit(Passage(f"doc::{pid}", "doc", "dmv", "License", "", [],
                                 "Replace a lost license with proof of identification.", [pid]),
                          "dense", i+1, 0.8, query) for i, pid in enumerate(["z", "a"])][:top_k]


def tuned_pipeline():
    dense = Dense()
    pipeline = HybridRetrievalPipeline(dense_retriever=dense, bm25_retriever=None,
        config=RetrievalConfig.for_mode("finetuned", segmentation="struct"))
    return pipeline, dense


def test_tuned_preserves_dense_ties_ignores_subqueries_and_never_calls_rrf(monkeypatch):
    pipeline, dense = tuned_pipeline()
    def forbidden(*args, **kwargs):
        raise AssertionError("Fine-tuned chain must bypass rank fusion")
    monkeypatch.setattr("retrieval.pipeline.reciprocal_rank_fusion", forbidden)
    result = pipeline.retrieve(RetrievalRequest("User: license", ["expansion"], top_k=2))
    assert [e.passage_id for e in result] == ["doc::z", "doc::a"]
    assert dense.queries == ["User: license"]
    assert all(e.rerank_score == 0.8 and e.rrf_score == 0 for e in result)


def test_tuned_optimization_adapter_keeps_dense_order_and_service_citations():
    pipeline, dense = tuned_pipeline()
    adapter = Part2RetrievalAdapter(pipeline, corpus_version="struct-v1")
    plan = QueryPlan(conversation_id="test", current_message="license", rewritten_query="license",
                     sub_queries=["expansion"], domain="dmv")
    snapshot = asyncio.run(adapter.retrieve(plan, adapter.default_options()))
    assert [e.passage_id for e in snapshot.evidence] == ["doc::z", "doc::a"]
    assert snapshot.rerank_model is None and snapshot.branch_counts["bm25"] == 0
    result = asyncio.run(run_native(pipeline, DialogueInput(conversation_id="test", current_message="license"),
                                   MockLLM(), corpus_version="struct-v1", top_k=1))
    assert result["response"]["citations"] == ["doc::z"]
    assert result["retrieval_chain"] == "finetuned" and result["segmentation"] == "struct"


def test_wrong_segmentation_is_rejected_before_model_loading(tmp_path):
    (tmp_path / "passages.jsonl").write_text("")
    (tmp_path / "corpus_metadata.json").write_text('{"segmentation":"section"}')
    with pytest.raises(RetrievalModelMismatch, match="segmentation"):
        HybridRetrievalPipeline.from_artifacts(artifact_dir=tmp_path,
            config=RetrievalConfig.for_mode("finetuned", segmentation="struct"))


def test_changed_checkpoint_rejects_existing_dense_index(tmp_path):
    from retrieval.dense import DenseRetriever
    from retrieval.corpus import PASSAGE_FINGERPRINT_VERSION
    (tmp_path / "model.safetensors").write_bytes(b"new weights")
    (tmp_path / "faiss.index").write_bytes(b"not reached")
    (tmp_path / "dense_metadata.json").write_text(json.dumps({
        "model_name": str(tmp_path), "fingerprint_version": PASSAGE_FINGERPRINT_VERSION,
        "local_model_fingerprint": "old-weight-hash",
    }))
    with pytest.raises(RetrievalModelMismatch, match="weights/config"):
        DenseRetriever.from_artifacts(tmp_path, str(tmp_path), passages=[Passage(
            "d::1", "d", "dmv", "Title", "", [], "Body", ["1"])])


def test_finetuned_context_keeps_last_four_visible_roles():
    dialogue = DialogueInput(conversation_id="history", current_message=" Current\nquestion ",
        dialogue_history=[{"role": "system", "text": "hidden"},
                          {"role": "user", "text": "oldest"},
                          {"role": "agent", "text": "ask one"},
                          {"role": "user", "text": "reply"},
                          {"role": "agent", "text": "ask two"}])
    assert finetuned_query(dialogue) == "Agent: ask one\nUser: reply\nAgent: ask two\nUser: Current question"
