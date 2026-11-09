import numpy as np

from customer_service_rag.retrieval.deduplication import EvidenceDeduplicator
from customer_service_rag.retrieval.schemas import FusedHit
from .helpers import passage


def test_exact_text_and_id_deduplication() -> None:
    a = FusedHit(passage("a", text="Same!"))
    b = FusedHit(passage("b", doc="d2", text="same"))
    assert [x.passage.passage_id for x in EvidenceDeduplicator().select([a, b], top_k=3)] == ["a"]


def test_per_document_cap() -> None:
    values = [FusedHit(passage("a", text="one")), FusedHit(passage("b", text="two"))]
    assert len(EvidenceDeduplicator(max_passages_per_doc=1).select(values, top_k=2)) == 1


def test_embedding_similarity_deduplication() -> None:
    passages = [passage("a", text="one"), passage("b", doc="d2", text="two")]
    candidates = [FusedHit(item) for item in passages]
    dedup = EvidenceDeduplicator(passages, np.array([[1.0, 0.0], [0.99, 0.01]]), similarity_threshold=0.9)
    assert len(dedup.select(candidates, top_k=2)) == 1
