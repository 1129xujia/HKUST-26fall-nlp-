"""Lazy local-only Transformers loaders shared by requests."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from customer_service_rag.assets import file_sha256


@dataclass
class Runtime:
    tokenizer: Any
    model: Any
    lock: threading.Lock
    torch: Any


@lru_cache(maxsize=4)
def load_runtime(
    directory: str, kind: str, repo_id: str, revision: str, num_threads: int, device: str = "cpu",
) -> Runtime:
    try:
        import torch
        from transformers import (
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError("Install local-model dependencies with pip install -e '.[local]'") from exc
    path = Path(directory)
    registry_file = path.parent / "registry.json"
    if not registry_file.exists():
        raise FileNotFoundError("Local model registry missing; run cs-rag download-models")
    entry = json.loads(registry_file.read_text(encoding="utf-8")).get(kind)
    if entry is None or entry["repo_id"] != repo_id or entry["revision"] != revision:
        raise ValueError(f"Pinned {kind} model does not match the downloaded registry")
    for record in entry["files"]:
        if file_sha256(path / record["path"]) != record["sha256"]:
            raise ValueError(f"Downloaded model checksum mismatch: {kind}/{record['path']}")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but this PyTorch installation has no usable CUDA device")
    torch.set_num_threads(num_threads)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model_class = {
        "llm": AutoModelForCausalLM,
        "embedding": AutoModel,
        "reranker": AutoModelForSequenceClassification,
    }[kind]
    model = model_class.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.float32 if device == "cpu" else torch.float16,
    ).to(device).eval()
    if kind == "llm":
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    return Runtime(tokenizer=tokenizer, model=model, lock=threading.Lock(), torch=torch)
