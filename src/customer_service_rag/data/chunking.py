"""Offset-preserving windows; official.py adds complete span-fragment coverage."""

from __future__ import annotations

from customer_service_rag.schemas import Passage


def chunk_document(
    *, doc_id: str, title: str, text: str, domain: str,
    chunk_size: int = 1200, overlap: int = 150, source: str = "multidoc2dial",
) -> list[Passage]:
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require 0 <= overlap < chunk_size")
    passages = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if chunk.strip():
            passages.append(Passage(
                doc_id=doc_id, passage_id=f"chars_{start}_{end}", title=title,
                text=chunk, domain=domain, char_start=start, char_end=end, source=source,
            ))
        if end == len(text):
            break
        start = end - overlap
    return passages
