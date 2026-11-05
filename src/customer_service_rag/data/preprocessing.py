from __future__ import annotations

from customer_service_rag.config import AppConfig
from customer_service_rag.data.io import load_examples, load_passages, write_jsonl


def prepare_mini(config: AppConfig) -> dict[str, int | str]:
    if config.data.source != "synthetic":
        raise ValueError("prepare_mini accepts synthetic data; use prepare_official for MultiDoc2Dial")
    passages = load_passages(config.resolve_path(config.data.passages_path))
    examples = load_examples(config.resolve_path(config.data.examples_path))
    available = {passage.citation_id for passage in passages}
    for example in examples:
        if example.source != "synthetic" or example.split != "mini":
            raise ValueError("prepare_mini accepts only explicitly synthetic mini examples")
        if not set(example.gold_citations).issubset(available):
            raise ValueError(f"Unknown gold citation in sample {example.sample_id}")
    output = config.resolve_path("data/processed")
    write_jsonl(output / "passages.jsonl", passages)
    write_jsonl(output / "mini.jsonl", examples)
    return {"passages": len(passages), "examples": len(examples), "output": str(output)}
