"""Prepare the exact main-config MiniLM models and official passage index here."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "vendor"))
sys.path.insert(0, str(ROOT / "main/src"))

from customer_service_rag.assets import MODEL_SPECS, download_file
from customer_service_rag.config import load_config
from customer_service_rag.data.official import prepare_official
from customer_service_rag.data.io import load_passages
from customer_service_rag.models.embeddings import build_embedding
from customer_service_rag.retrieval.indexing import build_index


def model_config():
    data_dir = HERE / "data"
    model_dir = HERE / "artifacts/models"
    return load_config(ROOT / "main/configs/integrated.yaml", overrides={
        "data": {
            "raw_dir": str(ROOT / "multidoc2dial/data/multidoc2dial"),
            "processed_dir": str(data_dir),
            "passages_path": str(data_dir / "passages.jsonl"),
            "examples_path": str(data_dir / "mini.jsonl"),
        },
        "models": {
            "embedding": {"local_path": str(model_dir / "embedding")},
            "reranker": {"local_path": str(model_dir / "reranker")},
        },
        "paths": {"index_dir": str(HERE / "artifacts/indexes/main_minilm")},
    })


def download_main_models(endpoint="https://hf-mirror.com"):
    directory = HERE / "artifacts/models"
    directory.mkdir(parents=True, exist_ok=True)
    registry_path = directory / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    allowed = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "vocab.txt", "sentence_bert_config.json",
               "config_sentence_transformers.json", "README.md", "LICENSE"}
    for kind in ("embedding", "reranker"):
        repo_id, revision = MODEL_SPECS[kind]
        url = f"{endpoint}/api/models/{repo_id}/revision/{revision}?blobs=true"
        with urllib.request.urlopen(url, timeout=30) as response:
            metadata = json.load(response)
        if metadata["sha"] != revision or metadata.get("cardData", {}).get("license") != "apache-2.0":
            raise ValueError(f"Unexpected {kind} revision or license")
        files = []
        for item in metadata["siblings"]:
            name = item["rfilename"]
            if name not in allowed:
                continue
            destination = directory / kind / name
            download_url = f"{endpoint}/{repo_id}/resolve/{revision}/{urllib.parse.quote(name)}"
            checksum = item.get("lfs", {}).get("sha256")
            record = download_file(download_url, destination, checksum)
            if not checksum:
                content = destination.read_bytes()
                git_hash = __import__("hashlib").sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
                if git_hash != item["blobId"]:
                    raise ValueError(f"Git blob checksum mismatch: {kind}/{name}")
            record.update(path=name, original_url=f"https://huggingface.co/{repo_id}/resolve/{revision}/{name}")
            files.append(record)
        if not (directory / kind / "model.safetensors").is_file():
            raise FileNotFoundError(f"Missing {kind} weights")
        registry[kind] = {"repo_id": repo_id, "revision": revision, "license": "Apache-2.0",
                          "path": (directory / kind).relative_to(HERE).as_posix(),
                          "endpoint": endpoint, "files": files}
        registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        print(f"Prepared main {kind}: {repo_id}@{revision}", flush=True)


if __name__ == "__main__":
    download_main_models()
    config = model_config()
    processed = prepare_official(config)
    print(f"Official preprocessing: {processed.get('passages', 'see manifest')} passages", flush=True)
    passages = load_passages(config.resolve_path(config.data.passages_path))
    embedding = build_embedding(config.models.embedding, config.project_root)
    manifest = build_index(passages, embedding, config.resolve_path(config.paths.index_dir))
    print(f"Main index: {manifest['passage_count']} passages, {manifest['embedding_model']}", flush=True)
