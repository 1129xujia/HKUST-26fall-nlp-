"""MultiDoc2Dial preprocessing with offset-exact, alternative evidence groups."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

from customer_service_rag.assets import file_sha256
from customer_service_rag.config import AppConfig
from customer_service_rag.data.chunking import chunk_document
from customer_service_rag.data.io import write_jsonl
from customer_service_rag.data.multidoc2dial import iter_dialogues, iter_documents
from customer_service_rag.schemas import (
    ChatMessage,
    DialogueInput,
    EvaluationExample,
    GroundingReference,
    Passage,
)


def _find(raw: Path, filename: str) -> Path:
    candidates = list(raw.rglob(filename))
    # Prefer the root distribution over any domain-specific copies.
    candidates.sort(key=lambda path: (len(path.parts), path.as_posix()))
    if not candidates:
        raise FileNotFoundError(f"{filename} not found under {raw}; run download-data")
    return candidates[0]


def _qualified(domain: str, doc_id: str) -> str:
    return f"{quote(domain, safe='')}/{quote(doc_id, safe='')}"


def _pieces(start: int, end: int, size: int) -> list[tuple[int, int]]:
    return [(offset, min(end, offset + size)) for offset in range(start, end, size)]


def prepare_official(config: AppConfig) -> dict:
    raw = config.resolve_path(config.data.raw_dir)
    destination = config.resolve_path(config.data.processed_dir)
    destination.mkdir(parents=True, exist_ok=True)
    doc_path = _find(raw, "multidoc2dial_doc.json")
    documents = list(iter_documents(doc_path))
    corpus: list[Passage] = []
    maps: dict[tuple[str, str], dict] = {}
    raw_lookup: dict[str, list[tuple[str, str]]] = defaultdict(list)
    mismatches = 0
    for doc in documents:
        domain, original_id, text = doc["domain"], doc["doc_id"], doc["text"]
        key = (domain, original_id)
        if key in maps:
            raise ValueError(f"Duplicate document key: {key}")
        doc_id = _qualified(*key)
        passages = chunk_document(
            doc_id=doc_id, title=doc["title"], text=text, domain=domain,
            chunk_size=config.data.chunk_size, overlap=config.data.chunk_overlap,
        )
        windows = {(p.char_start, p.char_end): p for p in passages}
        spans = {}
        for span_id, span in doc["spans"].items():
            start, end = int(span["start_sp"]), int(span["end_sp"])
            if not 0 <= start < end <= len(text):
                raise ValueError(f"Invalid span offset: {key}/{span_id}: {start}:{end}")
            if text[start:end] != span.get("text_sp", text[start:end]):
                mismatches += 1
            pieces = _pieces(start, end, config.data.chunk_size - config.data.chunk_overlap)
            for left, right in pieces:
                if not any(a <= left and b >= right for a, b in windows):
                    a = min(left, max(0, len(text) - config.data.chunk_size))
                    b = min(len(text), a + config.data.chunk_size)
                    windows[(a, b)] = Passage(
                        doc_id=doc_id, passage_id=f"chars_{a}_{b}", title=doc["title"],
                        text=text[a:b], domain=domain, char_start=a, char_end=b,
                        source="multidoc2dial", original_doc_id=original_id,
                    )
            spans[str(span_id)] = {"start": start, "end": end, "pieces": pieces}
        passages = [windows[window] for window in sorted(windows)]
        for passage in passages:
            passage.original_doc_id = original_id
            passage.span_ids = [
                span_id for span_id, span in spans.items()
                if passage.char_start < span["end"] and passage.char_end > span["start"]
            ]
        for span in spans.values():
            span["groups"] = [[
                p.citation_id for p in passages if p.char_start <= left and p.char_end >= right
            ] for left, right in span.pop("pieces")]
            if any(not group for group in span["groups"]):
                raise ValueError(f"Uncovered grounding span in {key}")
        maps[key] = {"doc_id": doc_id, "spans": spans}
        raw_lookup[original_id].append(key)
        corpus.extend(passages)

    outputs: dict[str, list[EvaluationExample]] = {
        name: [] for name in ["train", "dev", "validation", "mini"]
    }
    dialogue_splits: dict[str, str] = {}
    input_files = [doc_path]
    seen_dialogues: set[str] = set()
    for origin in ["train", "validation"]:
        path = _find(raw, f"multidoc2dial_dial_{origin}.json")
        input_files.append(path)
        for dialogue in iter_dialogues(path):
            domain, dial_id = dialogue["domain"], str(dialogue["dial_id"])
            identity = f"{domain}/{dial_id}"
            if identity in seen_dialogues:
                raise ValueError(f"Dialogue duplicated across official splits: {identity}")
            seen_dialogues.add(identity)
            value = int(hashlib.sha256(f"{config.seed}/{identity}".encode()).hexdigest()[:16], 16)
            split = "dev" if origin == "train" and value / 2**64 < config.data.dev_ratio else origin
            dialogue_splits[identity] = split
            # Opaque online IDs prevent the dataset domain from leaking through prompt metadata.
            conversation_id = "dialogue_" + hashlib.sha256(identity.encode()).hexdigest()[:20]
            turns = dialogue["turns"]
            ids = [int(turn["turn_id"]) for turn in turns]
            if ids != sorted(set(ids)):
                raise ValueError(f"Non-monotonic turn IDs: {identity}")
            history: list[ChatMessage] = []
            index = 0
            while index < len(turns):
                if turns[index]["role"] != "user":
                    history.append(ChatMessage(role="agent", text=turns[index]["utterance"]))
                    index += 1
                    continue
                user_start = index
                while index < len(turns) and turns[index]["role"] == "user":
                    index += 1
                user_turns = turns[user_start:index]
                agent_start = index
                while index < len(turns) and turns[index]["role"] == "agent":
                    index += 1
                agent_turns = turns[agent_start:index]
                if not agent_turns:
                    break
                references = []
                evidence_groups = []
                seen_refs = set()
                for turn in agent_turns:
                    for reference in turn.get("references", []):
                        original_doc_id, span_id = str(reference["doc_id"]), str(reference["id_sp"])
                        key = (domain, original_doc_id)
                        if key not in maps:
                            alternatives = raw_lookup.get(original_doc_id, [])
                            if len(alternatives) != 1:
                                raise ValueError(f"Unknown or ambiguous reference: {identity}/{reference}")
                            key = alternatives[0]
                        if (key, span_id) in seen_refs:
                            continue
                        seen_refs.add((key, span_id))
                        span = maps[key]["spans"][span_id]
                        evidence_groups.extend(span["groups"])
                        references.append(GroundingReference(
                            doc_id=maps[key]["doc_id"], original_doc_id=original_doc_id,
                            span_ids=[span_id], label=reference.get("label"),
                            char_start=span["start"], char_end=span["end"],
                        ))
                unique_groups = []
                for group in evidence_groups:
                    if group not in unique_groups:
                        unique_groups.append(group)
                example = EvaluationExample(
                    sample_id=f"{identity}/u{user_turns[0]['turn_id']}",
                    split=split,
                    dialogue=DialogueInput(
                        conversation_id=conversation_id,
                        current_message="\n".join(turn["utterance"] for turn in user_turns),
                        dialogue_history=list(history),
                    ),
                    reference_answers=["\n".join(turn["utterance"] for turn in agent_turns)],
                    gold_citations=sorted({key for group in unique_groups for key in group}),
                    gold_evidence_groups=unique_groups, grounding=references,
                    source="multidoc2dial", domain=domain, origin_split=origin,
                    user_turn_ids=[int(turn["turn_id"]) for turn in user_turns],
                )
                outputs[split].append(example)
                history.extend(
                    ChatMessage(role=turn["role"], text=turn["utterance"])
                    for turn in [*user_turns, *agent_turns]
                )
    # The development mini never draws from the untouched official validation set.
    by_domain = defaultdict(list)
    for example in outputs["dev"]:
        if example.gold_evidence_groups:
            by_domain[example.domain].append(example)
    for values in by_domain.values():
        values.sort(key=lambda example: hashlib.sha256(
            f"{config.seed}/{example.sample_id}".encode()
        ).hexdigest())
    offset = 0
    while len(outputs["mini"]) < config.data.mini_size:
        selected = [by_domain[domain][offset] for domain in sorted(by_domain)
                    if offset < len(by_domain[domain])]
        if not selected:
            break
        for example in selected:
            if len(outputs["mini"]) < config.data.mini_size:
                outputs["mini"].append(example.model_copy(update={"split": "mini"}, deep=True))
        offset += 1
    write_jsonl(destination / "passages.jsonl", corpus)
    for split, examples in outputs.items():
        write_jsonl(destination / f"{split}.jsonl", examples)
    grounding_map = {f"{domain}/{doc_id}": values for (domain, doc_id), values in maps.items()}
    (destination / "grounding_map.json").write_text(
        json.dumps(grounding_map, ensure_ascii=False), encoding="utf-8",
    )
    manifest = {
        "dataset": "MultiDoc2Dial", "version": "1.0", "preprocessing_version": "offset-v1",
        "documents": len(documents), "passages": len(corpus),
        "examples": {name: len(examples) for name, examples in outputs.items()},
        "domains": sorted(by_domain), "seed": config.seed, "dev_ratio": config.data.dev_ratio,
        "chunk_size": config.data.chunk_size, "chunk_overlap": config.data.chunk_overlap,
        "span_text_mismatches": mismatches, "unmapped_references": 0,
        "files": {path.name: file_sha256(path) for path in input_files},
        "dialogue_splits": dialogue_splits,
        "protocol": "Train is split by dialogue into train/dev; official validation is untouched. "
                    "Consecutive user/agent turns are merged into one request/response. "
                    "Long spans require all fragment groups; overlapping windows are alternatives.",
        "official_test": "Dummy file, excluded from labeled evaluation.",
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return {key: value for key, value in manifest.items() if key != "dialogue_splits"}
