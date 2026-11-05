from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import BaseModel

from customer_service_rag.schemas import EvaluationExample, Passage

T = TypeVar("T", bound=BaseModel)


def read_jsonl(path: Path, model: type[T]) -> list[T]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(model.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"Invalid {model.__name__} at {path}:{line_number}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n"
        for record in records
    )
    path.write_text(content, encoding="utf-8")


def load_passages(path: Path) -> list[Passage]:
    passages = read_jsonl(path, Passage)
    ids = [passage.citation_id for passage in passages]
    if not passages:
        raise ValueError(f"Passage corpus is empty: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate passage citation IDs in {path}")
    return passages


def load_examples(path: Path) -> list[EvaluationExample]:
    examples = read_jsonl(path, EvaluationExample)
    ids = [example.sample_id for example in examples]
    if not examples:
        raise ValueError(f"Dialogue dataset is empty: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return examples
