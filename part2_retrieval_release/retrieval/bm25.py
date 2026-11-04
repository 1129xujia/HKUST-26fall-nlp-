"""BM25 lexical retrieval."""

import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from .corpus import PASSAGE_FINGERPRINT_VERSION, load_passages_jsonl, passage_fingerprint
from .exceptions import RetrievalIndexNotFound, RetrievalModelMismatch
from .schemas import Passage, SearchHit

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
QUERY_ROLE_TOKENS = {"user", "agent"}


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    def __init__(self, passages: list[Passage], index: BM25Okapi) -> None:
        self.passages, self.index = passages, index

    @classmethod
    def build(cls, passages: list[Passage]) -> "BM25Retriever":
        if not passages:
            raise ValueError("cannot build BM25 index from an empty corpus")
        return cls(passages, BM25Okapi([tokenize(item.index_text) for item in passages]))

    def save(self, artifact_dir: Path | str) -> None:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with (artifact_dir / "bm25.pkl").open("wb") as stream:
            pickle.dump(self.index, stream, protocol=pickle.HIGHEST_PROTOCOL)
        metadata = {
            "passage_count": len(self.passages),
            "fingerprint_version": PASSAGE_FINGERPRINT_VERSION,
            "passage_fingerprint": passage_fingerprint(self.passages),
        }
        (artifact_dir / "bm25_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def from_artifacts(cls, artifact_dir: Path | str, *, passages: list[Passage] | None = None) -> "BM25Retriever":
        artifact_dir = Path(artifact_dir)
        index_path, metadata_path = artifact_dir / "bm25.pkl", artifact_dir / "bm25_metadata.json"
        if not index_path.exists() or not metadata_path.exists():
            raise RetrievalIndexNotFound(f"Missing BM25 artifacts in {artifact_dir}")
        passages = passages or load_passages_jsonl(artifact_dir / "passages.jsonl")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint_version") != PASSAGE_FINGERPRINT_VERSION:
            raise RetrievalModelMismatch("BM25 fingerprint version is stale; rebuild indexes")
        if metadata.get("passage_count") != len(passages) or metadata.get("passage_fingerprint") != passage_fingerprint(passages):
            raise RetrievalModelMismatch("BM25 index was built from a different corpus")
        with index_path.open("rb") as stream:
            return cls(passages, pickle.load(stream))

    def search(self, query: str, *, domain: str | None = None, top_k: int = 20) -> list[SearchHit]:
        if top_k <= 0:
            return []
        # Role labels are evaluation/graph serialization markers, not search intent.
        tokens = [token for token in tokenize(query) if token not in QUERY_ROLE_TOKENS]
        if not tokens:
            return []
        scores = self.index.get_scores(tokens)
        hits: list[SearchHit] = []
        for row in sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True):
            item = self.passages[row]
            if domain and item.domain != domain:
                continue
            if not any(token in self.index.doc_freqs[row] for token in tokens):
                continue
            hits.append(SearchHit(item, "bm25", len(hits) + 1, float(scores[row]), query))
            if len(hits) >= top_k:
                break
        return hits
