"""Build and load section or official-structure corpora for every retriever."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from .schemas import Passage

PASSAGE_FINGERPRINT_VERSION = 2
TITLE_SECTION_MODES = {"independent", "merge", "metadata"}
SEGMENTATION_MODES = {"section", "struct"}
OFFICIAL_STRUCT_SOURCE_REVISION = "2456a5609b447e1a12b716f4d658728f346f3041"


def _iter_struct_passages(payload: dict[str, Any]) -> Iterable[Passage]:
    """复刻官方 split_text_section：相邻 span 同 section 或同标题就合并。"""
    for domain, documents in payload.get("doc_data", payload).items():
        for doc_id, document in documents.items():
            groups: list[list[tuple[str, dict[str, Any]]]] = []
            previous_section = previous_title = None
            # 官方按 JSON 中 span 的原始顺序遍历，不对所有同名标题做全局合并。
            for span_id, span in document.get("spans", {}).items():
                if not groups or not (previous_section == span["id_sec"] or previous_title == span["title"].strip()):
                    groups.append([])
                groups[-1].append((span_id, span))
                previous_section, previous_title = span["id_sec"], span["title"].strip()
            for ordinal, members in enumerate(groups):
                first, last = members[0][1], members[-1][1]
                clean_title = lambda value: str(value).replace("/", "-").rsplit("#")[0]
                parent_path = [clean_title(document.get("title", ""))]
                parents = last.get("parent_titles", [])
                if len(parents) > 1:
                    parent_path = [clean_title(p["text"]) for p in parents]
                body = " ".join(s["text_sp"] for _, s in members).replace("\n", " ")
                text = (" / ".join(parent_path) + " // " + body).replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
                yield Passage(
                    passage_id=f"{doc_id}::structure_{ordinal}", doc_id=str(doc_id), domain=str(domain),
                    title=str(document.get("title", "")), section_title=first.get("title", "").strip(),
                    parent_titles=[str(p.get("text", p.get("title", ""))).strip() for p in first.get("parent_titles", [])],
                    text=text, span_ids=[str(s.get("id_sp", sid)) for sid, s in members],
                    start=min(int(s["start_sp"]) for _, s in members),
                    end=max(int(s["end_sp"]) for _, s in members),
                )


def _normalise_parent_titles(values: Any) -> list[str]:
    result: list[str] = []
    for item in values or []:
        value = (item.get("title") or item.get("text") or "") if isinstance(item, dict) else str(item)
        if value.strip():
            result.append(value.strip())
    return result


def _ordered_sections(document: dict[str, Any]) -> list[tuple[str, list[tuple[str, dict[str, Any]]]]]:
    ordered = sorted(
        document.get("spans", {}).items(),
        key=lambda item: (int(item[1].get("start_sp", 0)), str(item[0])),
    )
    sections: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for span_id, span in ordered:
        sections.setdefault(str(span.get("id_sec", span_id)), []).append((str(span_id), span))
    return list(sections.items())


def _normalise_title(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _heading_text(members: list[tuple[str, dict[str, Any]]]) -> str:
    first = members[0][1]
    return str(first.get("title") or first.get("text_sec") or "").strip()


def _title_targets(
    sections: list[tuple[str, list[tuple[str, dict[str, Any]]]]],
) -> dict[str, str]:
    """Map a title section to its following body only when their titles agree."""
    targets: dict[str, str] = {}
    for index, (section_id, members) in enumerate(sections):
        if not section_id.startswith("t_"):
            continue
        heading = _normalise_title(_heading_text(members))
        for target_id, target_members in sections[index + 1 :]:
            if target_id.startswith("t_"):
                break
            target_title = _normalise_title(target_members[0][1].get("title"))
            if heading and heading == target_title:
                targets[section_id] = target_id
            break
    return targets


def build_title_section_map(dataset_path: Path | str) -> dict[str, str]:
    """Return reliable full passage-ID mappings from title sections to body sections."""
    with Path(dataset_path).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    result: dict[str, str] = {}
    for documents in payload.get("doc_data", payload).values():
        for doc_id, document in documents.items():
            for title_id, target_id in _title_targets(_ordered_sections(document)).items():
                result[f"{doc_id}::{title_id}"] = f"{doc_id}::{target_id}"
    return result


def title_section_stats(dataset_path: Path | str) -> dict[str, int]:
    """Count unique title sections and how many have a reliable body target."""
    with Path(dataset_path).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    total = 0
    mapped = 0
    for documents in payload.get("doc_data", payload).values():
        for document in documents.values():
            sections = _ordered_sections(document)
            total += sum(section_id.startswith("t_") for section_id, _ in sections)
            mapped += len(_title_targets(sections))
    return {
        "original_title_section_count": total,
        "mapped_title_section_count": mapped,
        "unmapped_title_section_count": total - mapped,
    }


def iter_passages(
    dataset_path: Path | str,
    *,
    title_section_mode: str = "independent",
    segmentation: str = "section",
) -> Iterable[Passage]:
    if segmentation not in SEGMENTATION_MODES:
        raise ValueError("segmentation must be 'section' or 'struct'")
    if title_section_mode not in TITLE_SECTION_MODES:
        raise ValueError(
            f"title_section_mode must be one of {sorted(TITLE_SECTION_MODES)}"
        )
    with Path(dataset_path).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if segmentation == "struct":
        if title_section_mode != "independent":
            raise ValueError("title_section_mode applies only to section segmentation")
        yield from _iter_struct_passages(payload)
        return
    for domain, documents in payload.get("doc_data", payload).items():
        for doc_id, document in documents.items():
            sections = _ordered_sections(document)
            title_targets = _title_targets(sections)
            headings_by_target: dict[str, list[list[tuple[str, dict[str, Any]]]]] = {}
            for title_id, target_id in title_targets.items():
                title_members = next(members for section_id, members in sections if section_id == title_id)
                headings_by_target.setdefault(target_id, []).append(title_members)

            for section_id, members in sections:
                if title_section_mode != "independent" and section_id.startswith("t_"):
                    continue
                first = members[0][1]
                start = int(first.get("start_sec", first.get("start_sp", 0)))
                end = int(first.get("end_sec", first.get("end_sp", start)))
                text = str(first.get("text_sec") or document.get("doc_text", "")[start:end]).strip()
                if not text:
                    continue
                section_title = str(first.get("title", "")).strip()
                parent_titles = _normalise_parent_titles(first.get("parent_titles"))
                span_ids = [str(span.get("id_sp", span_id)) for span_id, span in members]

                attached_headings = headings_by_target.get(section_id, [])
                if title_section_mode != "independent" and attached_headings:
                    heading_titles = [title for heading in attached_headings if (title := _heading_text(heading))]
                    heading_span_ids = [
                        str(span.get("id_sp", span_id))
                        for heading in attached_headings
                        for span_id, span in heading
                    ]
                    heading_starts = [
                        int(heading[0][1].get("start_sec", heading[0][1].get("start_sp", start)))
                        for heading in attached_headings
                    ]
                    if heading_titles:
                        section_title = " > ".join(heading_titles)
                        if title_section_mode == "merge":
                            text = "\n\n".join([*heading_titles, text])
                    span_ids = [*heading_span_ids, *span_ids]
                    start = min([start, *heading_starts])
                yield Passage(
                    passage_id=f"{doc_id}::{section_id}",
                    doc_id=str(doc_id),
                    domain=str(domain),
                    title=str(document.get("title", "")),
                    section_title=section_title,
                    parent_titles=parent_titles,
                    text=text,
                    span_ids=span_ids,
                    start=start,
                    end=end,
                )


def write_passages_jsonl(passages: Iterable[Passage] | Path | str, output_path: Path | str) -> int:
    if isinstance(passages, (Path, str)):
        passages = iter_passages(passages)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for passage in passages:
                stream.write(json.dumps(passage.to_dict(), ensure_ascii=False) + "\n")
                count += 1
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count


def load_passages_jsonl(path: Path | str) -> list[Passage]:
    with Path(path).open(encoding="utf-8") as stream:
        return [Passage(**json.loads(line)) for line in stream if line.strip()]


def passage_fingerprint(passages: Iterable[Passage]) -> str:
    digest = hashlib.sha256(f"passage-fingerprint-v{PASSAGE_FINGERPRINT_VERSION}\n".encode())
    for passage in passages:
        digest.update(json.dumps(passage.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()
