"""Read official JSON containers without guessing grounding/chunk alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def _container(path: Path, key: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping named {key!r}")
    return value


def iter_documents(path: Path) -> Iterator[dict[str, Any]]:
    for domain, documents in _container(path, "doc_data").items():
        if not isinstance(documents, dict):
            raise ValueError(f"Expected document mapping for domain {domain!r}")
        for doc_id, document in documents.items():
            if not isinstance(document, dict) or not isinstance(document.get("doc_text"), str):
                raise ValueError(f"Missing doc_text for {domain}/{doc_id}")
            yield {
                "doc_id": doc_id,
                "domain": domain,
                "title": document.get("title", doc_id),
                "text": document["doc_text"],
                "spans": document.get("spans", {}),
            }


def _dialogues(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from _dialogues(item)
    elif isinstance(value, dict):
        if "dial_id" in value and isinstance(value.get("turns"), list):
            yield value
        else:
            for item in value.values():
                yield from _dialogues(item)
    else:
        raise ValueError("Expected a dialogue list or document-to-dialogues mapping")


def iter_dialogues(path: Path) -> Iterator[dict[str, Any]]:
    for domain, dialogues in _container(path, "dial_data").items():
        for dialogue in _dialogues(dialogues):
            yield {"domain": domain, **dialogue}


def inspect_dataset(documents_path: Path, dialogues_path: Path) -> dict[str, Any]:
    documents = list(iter_documents(documents_path))
    dialogues = list(iter_dialogues(dialogues_path))
    if not documents or not dialogues:
        raise ValueError("No documents or dialogues found; check dataset version and JSON layout")
    return {
        "documents": len(documents),
        "dialogues": len(dialogues),
        "turns": sum(len(item["turns"]) for item in dialogues),
        "domains": sorted({item["domain"] for item in documents}),
        "note": "Raw-container inspection. The preprocess command separately aligns grounding spans.",
    }
