"""Explicit, checksum-verified downloads into the project workspace."""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DATASET_URL = "https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip"
MODEL_SPECS = {
    "llm": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "embedding": (
        "sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    ),
    "reranker": (
        "cross-encoder/ms-marco-MiniLM-L6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, path: Path, expected_sha256: str | None = None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and expected_sha256 and file_sha256(path) == expected_sha256:
        return {"file": path.name, "sha256": expected_sha256, "bytes": path.stat().st_size}
    temporary = path.with_name(path.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "customer-service-rag/0.1"})
    print(f"Downloading {path.name}", file=sys.stderr, flush=True)
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        expected_size = response.headers.get("Content-Length")
        total = 0
        for block in iter(lambda: response.read(2**20), b""):
            output.write(block)
            total += len(block)
        if expected_size is not None and total != int(expected_size):
            raise ValueError(f"Incomplete download: {path.name}")
    checksum = file_sha256(temporary)
    if expected_sha256 is not None and checksum != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path.name}")
    temporary.replace(path)
    return {"file": path.name, "sha256": checksum, "bytes": path.stat().st_size, "url": url}


def download_dataset(root: Path) -> dict:
    directory = root / "data/raw/multidoc2dial"
    archive = directory / "multidoc2dial.zip"
    if not archive.is_file():
        download_file(DATASET_URL, archive)
    extracted = []
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            if item.is_dir():
                continue
            name = Path(item.filename).name
            if "__MACOSX" in Path(item.filename).parts or name.startswith("._"):
                continue
            if not (name.endswith(".json") or name.lower().startswith(("readme", "license"))):
                continue
            target = (directory / item.filename).resolve()
            if not target.is_relative_to(directory.resolve()):
                raise ValueError("Archive member escapes the raw-data directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(item) as source, target.open("wb") as destination:
                destination.write(source.read())
            extracted.append({
                "path": target.relative_to(root).as_posix(),
                "sha256": file_sha256(target), "bytes": target.stat().st_size,
            })
    license_url = "https://raw.githubusercontent.com/IBM/multidoc2dial/main/LICENSE"
    license_path = directory / "LICENSE"
    if not license_path.is_file():
        download_file(license_url, license_path)
    manifest = {
        "dataset": "MultiDoc2Dial", "version": "1.0", "source_url": DATASET_URL,
        "archive_sha256": file_sha256(archive),
        "license": "Apache-2.0",
        "license_source": "https://huggingface.co/datasets/IBM/multidoc2dial",
        "repository_license_url": license_url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": extracted,
        "test_split": "Official v1.0 test JSON is a dummy file; not used for labeled evaluation.",
    }
    (directory / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    return manifest


def download_models(
    root: Path, endpoint: str = "https://hf-mirror.com", *, include_llm: bool = False,
) -> dict:
    directory = root / "artifacts/models"
    directory.mkdir(parents=True, exist_ok=True)
    registry_path = directory / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    allowed = {
        "config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "vocab.txt",
        "merges.txt", "sentence_bert_config.json", "config_sentence_transformers.json",
        "README.md", "LICENSE",
    }
    for kind, (repo_id, revision) in MODEL_SPECS.items():
        if kind == "llm" and not include_llm:
            continue
        url = f"{endpoint}/api/models/{repo_id}/revision/{revision}?blobs=true"
        with urllib.request.urlopen(url, timeout=30) as response:
            metadata = json.load(response)
        if metadata["sha"] != revision or metadata.get("cardData", {}).get("license") != "apache-2.0":
            raise ValueError(f"Unexpected model revision or license: {repo_id}")
        files = []
        for item in metadata["siblings"]:
            name = item["rfilename"]
            if name not in allowed:
                continue
            destination = directory / kind / name
            original_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{name}"
            download_url = f"{endpoint}/{repo_id}/resolve/{revision}/{urllib.parse.quote(name)}"
            checksum = item.get("lfs", {}).get("sha256")
            record = download_file(download_url, destination, checksum)
            if not checksum:
                content = destination.read_bytes()
                git_hash = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
                if git_hash != item["blobId"]:
                    raise ValueError(f"Git blob checksum mismatch: {repo_id}/{name}")
            record.update(path=name, original_url=original_url)
            files.append(record)
        if not (directory / kind / "model.safetensors").is_file():
            raise ValueError(f"Missing safe model weights for {kind}")
        registry[kind] = {
            "repo_id": repo_id, "revision": revision, "license": "Apache-2.0",
            "path": (directory / kind).relative_to(root).as_posix(),
            "endpoint": endpoint, "files": files,
        }
        registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry
