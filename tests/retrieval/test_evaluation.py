import pytest
import json

from customer_service_rag.retrieval.evaluation import RetrievalEvalSample, evaluate_candidate_recall, evaluate_pipeline
from customer_service_rag.retrieval.schemas import Evidence, Passage, SearchHit


class StubPipeline:
    def retrieve(self, request):
        return [Evidence("user-gold", "d", "dmv", "T", "S", "text", ["1"], ["bm25"], request.query)]


class StubRetriever:
    def search(self, query, *, domain=None, top_k=20):
        passages = [
            Passage("irrelevant", "d", "dmv", "T", "S", [], "text", ["1"], 0, 4),
            Passage("user-gold", "d", "dmv", "T", "S", [], "text", ["2"], 5, 9),
        ]
        return [
            SearchHit(passage, "bm25", rank, 1.0 / rank, query)
            for rank, passage in enumerate(passages[:top_k], 1)
        ]


def test_evaluation_metrics() -> None:
    sample = RetrievalEvalSample("s", "dmv", "q", ["user-gold"], ["agent-gold"])
    result = evaluate_pipeline(StubPipeline(), [sample], cutoffs=(1,))
    assert result["summary"]["hit_rate@1"] == 1.0
    assert result["summary"]["recall@1"] == 0.5
    assert result["summary"]["user_gold_hit_rate@1"] == 1.0
    assert result["summary"]["user_gold_recall@1"] == 1.0
    assert result["summary"]["agent_gold_hit_rate@1"] == 0.0
    assert result["summary"]["agent_gold_recall@1"] == 0.0
    assert "hit_recall@1" not in result["summary"]
    assert "evidence_coverage@1" not in result["summary"]
    assert result["samples"][0]["gold_passage_ids"] == ["agent-gold", "user-gold"]


def test_evaluation_rejects_empty_samples() -> None:
    with pytest.raises(ValueError):
        evaluate_pipeline(StubPipeline(), [])


def test_evaluation_rejects_bad_cutoff() -> None:
    with pytest.raises(ValueError):
        evaluate_pipeline(StubPipeline(), [RetrievalEvalSample("s", "dmv", "q", ["g"], [])], cutoffs=(0,))


def test_title_section_diagnostics_do_not_replace_official_metrics() -> None:
    sample = RetrievalEvalSample("s", "dmv", "q", ["title-gold"], [])
    result = evaluate_pipeline(
        StubPipeline(),
        [sample],
        cutoffs=(1,),
        title_section_map={"title-gold": "user-gold"},
    )
    summary = result["summary"]
    assert summary["hit_rate@1"] == 0.0
    assert summary["recall@1"] == 0.0
    assert summary["merged_title_hit_rate@1"] == 1.0
    assert summary["merged_title_recall@1"] == 1.0
    assert summary["merged_title_mrr"] == 1.0
    assert result["samples"][0]["gold_passage_ids"] == ["title-gold"]
    assert result["samples"][0]["merged_title_gold_passage_ids"] == ["user-gold"]


def test_candidate_recall_exposes_late_and_missing_gold() -> None:
    sample = RetrievalEvalSample("s", "dmv", "q", ["user-gold"], ["agent-gold"])
    result = evaluate_candidate_recall(StubRetriever(), [sample], cutoffs=(1, 2))
    summary = result["summary"]
    assert summary["recall@1"] == 0.0
    assert summary["recall@2"] == 0.5
    assert summary["user_gold_recall@2"] == 1.0
    assert summary["agent_gold_recall@2"] == 0.0
    assert summary["no_hit_rate@1"] == 1.0
    assert result["samples"][0]["gold_ranks"] == {
        "agent-gold": None,
        "user-gold": 2,
    }


def test_query_modes_can_remove_agent_text_and_put_current_first(tmp_path) -> None:
    documents = {
        "doc_data": {"dmv": {"doc": {"spans": {"1": {"id_sec": "sec"}}}}}
    }
    dialogues = {
        "dial_data": {"dmv": [{
            "dial_id": "dialogue",
            "turns": [
                {"turn_id": 1, "role": "user", "utterance": "old topic", "references": []},
                {"turn_id": 2, "role": "agent", "utterance": "long agent answer", "references": []},
                {"turn_id": 3, "role": "user", "utterance": "current question", "references": [{"doc_id": "doc", "id_sp": "1"}]},
            ],
        }]}
    }
    document_path = tmp_path / "documents.json"
    dialogue_path = tmp_path / "dialogues.json"
    document_path.write_text(json.dumps(documents), encoding="utf-8")
    dialogue_path.write_text(json.dumps(dialogues), encoding="utf-8")

    from customer_service_rag.retrieval.evaluation import build_validation_samples

    user_history = build_validation_samples(
        document_path=document_path, dialogue_path=dialogue_path,
        query_mode="user_history", history_turns=4,
    )
    current_first = build_validation_samples(
        document_path=document_path, dialogue_path=dialogue_path,
        query_mode="current_first_user_history", history_turns=4,
    )
    assert user_history[0].query == "old topic\ncurrent question"
    assert current_first[0].query == "current question\nold topic"
