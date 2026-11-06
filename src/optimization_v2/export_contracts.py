"""Export a self-contained JSON Schema bundle (all references resolve locally)."""

import json
from pathlib import Path

from pydantic.json_schema import models_json_schema

from .schemas import OptimizationAction, OptimizationInput, OptimizationResult


def build_schema():
    entries, schema = models_json_schema([
        (OptimizationInput, "validation"),
        (OptimizationAction, "validation"),
        (OptimizationResult, "serialization"),
    ], title="Optimization v2 public contracts")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$ref"] = entries[(OptimizationInput, "validation")]["$ref"]
    return schema


if __name__ == "__main__":
    destination = Path(__file__).with_name("contracts.schema.json")
    destination.write_text(
        json.dumps(build_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(destination)
