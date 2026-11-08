"""Identify local encoders by contents, rather than by mutable directory names."""

import hashlib
from pathlib import Path


def local_model_fingerprint(model_name: str) -> str | None:
    root = Path(model_name)
    if not root.is_dir():
        return None
    weights = sorted([*root.glob("*.safetensors"), *root.glob("pytorch_model*.bin")])
    if not weights:
        raise ValueError(f"No encoder weights in {root}; download the full model first")
    for path in weights:
        with path.open("rb") as stream:
            if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"{path} is a Git LFS pointer; run git lfs pull")
    files = sorted(set(weights + list(root.rglob("*.json")) + list(root.glob("*.txt"))))
    digest = hashlib.sha256()
    for path in files:
        # Provenance/readme changes do not change the encoder.
        if path.name == "experiment_provenance.json":
            continue
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
