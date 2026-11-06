import json
import pytest

from retrieval.corpus import (
    build_title_section_map,
    iter_passages,
    passage_fingerprint,
    title_section_stats,
    write_passages_jsonl,
)
from tests.helpers import passage


def test_fingerprint_changes_with_body() -> None:
    assert passage_fingerprint([passage(text="a")]) != passage_fingerprint([passage(text="b")])


def test_jsonl_roundtrip_write(tmp_path) -> None:
    output = tmp_path / "p.jsonl"
    assert write_passages_jsonl([passage()], output) == 1
    assert json.loads(output.read_text())["passage_id"] == "p1"


def test_iter_groups_spans_by_section_and_normalizes_parent_titles(tmp_path) -> None:
    payload = {"doc_data": {"dmv": {"doc": {"title": "Doc", "doc_text": "hello world", "spans": {
        "1": {"id_sp": "1", "tag": "u", "id_sec": "s", "start_sp": 0, "end_sp": 5, "start_sec": 0, "end_sec": 11, "text_sec": "hello world", "title": "S", "parent_titles": [{"title": "P"}]},
        "2": {"id_sp": "2", "tag": "u", "id_sec": "s", "start_sp": 6, "end_sp": 11, "start_sec": 0, "end_sec": 11, "text_sec": "hello world", "title": "S", "parent_titles": [{"title": "P"}]}
    }}}}}
    source = tmp_path / "d.json"
    source.write_text(json.dumps(payload))
    result = list(iter_passages(source))
    assert len(result) == 1
    assert result[0].passage_id == "doc::s"
    assert result[0].span_ids == ["1", "2"]
    assert result[0].parent_titles == ["P"]


def test_heading_sections_are_kept_for_gold_reference_alignment(tmp_path) -> None:
    source = tmp_path / "d.json"
    source.write_text(json.dumps({"doc_data": {"dmv": {"d": {
        "title": "Heading", "spans": {"1": {
            "tag": "h2", "id_sec": "title", "id_sp": "1",
            "text_sec": "Heading", "start_sp": 0, "end_sp": 7,
        }}
    }}}}))
    assert list(iter_passages(source))[0].passage_id == "d::title"


def test_title_section_modes_use_order_and_title_instead_of_numeric_suffix(tmp_path) -> None:
    source = tmp_path / "d.json"
    source.write_text(json.dumps({"doc_data": {"dmv": {"d": {
        "title": "Document", "spans": {
            "1": {
                "tag": "h2", "id_sec": "t_7", "id_sp": "1",
                "text_sec": "Correct heading", "title": "Correct heading",
                "start_sp": 0, "end_sp": 15,
            },
            "2": {
                "tag": "u", "id_sec": "8", "id_sp": "2",
                "text_sec": "Body evidence", "title": "Correct heading",
                "start_sp": 16, "end_sp": 29,
            },
        }
    }}}}), encoding="utf-8")

    independent = list(iter_passages(source))
    merged = list(iter_passages(source, title_section_mode="merge"))
    metadata = list(iter_passages(source, title_section_mode="metadata"))

    assert [item.passage_id for item in independent] == ["d::t_7", "d::8"]
    assert [item.passage_id for item in merged] == ["d::8"]
    assert merged[0].text == "Correct heading\n\nBody evidence"
    assert merged[0].span_ids == ["1", "2"]
    assert metadata[0].text == "Body evidence"
    assert metadata[0].section_title == "Correct heading"
    assert build_title_section_map(source) == {"d::t_7": "d::8"}
    assert title_section_stats(source) == {
        "original_title_section_count": 1,
        "mapped_title_section_count": 1,
        "unmapped_title_section_count": 0,
    }


def test_invalid_title_section_mode_is_rejected(tmp_path) -> None:
    source = tmp_path / "d.json"
    source.write_text(json.dumps({"doc_data": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_passages(source, title_section_mode="unknown"))
