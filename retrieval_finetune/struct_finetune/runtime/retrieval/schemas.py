"""Shared input, output, and internal retrieval data structures."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    sub_queries: list[str] = field(default_factory=list)
    domain: str | None = None
    top_k: int = 6
    trace_id: str | None = None


@dataclass(frozen=True)
class Passage:
    passage_id: str
    doc_id: str
    domain: str
    title: str
    section_title: str
    parent_titles: list[str]
    text: str
    span_ids: list[str]
    start: int | None = None
    end: int | None = None

    @property
    def index_text(self) -> str:
        parts = [
            self.title,
            " > ".join(self.parent_titles),
            self.section_title,
            self.text,
        ]
        return "\n".join(part.strip() for part in parts if part and part.strip())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchHit:
    passage: Passage
    retriever: str
    rank: int
    raw_score: float
    matched_query: str


@dataclass
class FusedHit:
    passage: Passage
    source_retrievers: list[str] = field(default_factory=list)
    matched_queries: list[str] = field(default_factory=list)
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float = 0.0


@dataclass(frozen=True)
class Evidence:
    passage_id: str
    doc_id: str
    domain: str
    title: str
    section_title: str
    text: str
    span_ids: list[str]
    source_retrievers: list[str]
    matched_query: str
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
