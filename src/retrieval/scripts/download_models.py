"""Download pinned public models, resume transfers and switch sources on failure."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


MODELS = [
    {
        "repo": "BAAI/bge-base-en-v1.5",
        "revision": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        "weight_sha256": "c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7",
        "files": ["config.json", "config_sentence_transformers.json", "modules.json", "sentence_bert_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "1_Pooling/config.json", "model.safetensors"],
    },
    {
        "repo": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        "weight_sha256": "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae",
        "files": ["config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "model.safetensors"],
    },
    {
        "repo": "alibaba-nlp-community/gte-base-en-v1.5",
        "revision": "532bc29f1fcb9f3d446568efdaa2c66af9255cf9",
        "weight_sha256": "c30d96b914af1a141db13075863e666dee17daa5f844d506e07c88cd9ab12d2b",
        "files": ["config.json", "modules.json", "sentence_bert_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "1_Pooling/config.json", "model.safetensors"],
    },
    {
        "repo": "intfloat/e5-base-v2",
        "revision": "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        "weight_sha256": "d0d559c47d5f71b1d280b13b62a2657f3e3bc70c0786f9ab91a36545e6a8f693",
        "files": ["config.json", "modules.json", "sentence_bert_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "1_Pooling/config.json", "model.safetensors"],
    },
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("HF_HOME", ".model_cache")))
    parser.add_argument("--mirror-first", action="store_true")
    args = parser.parse_args()
    sources = ["https://huggingface.co", "https://hf-mirror.com"]
    if args.mirror_first:
        sources.reverse()
    manifest = []
    for spec in MODELS:
        repo_dir = args.cache_dir / "hub" / ("models--" + spec["repo"].replace("/", "--"))
        snapshot = repo_dir / "snapshots" / spec["revision"]
        files = {}
        for name in spec["files"]:
            destination = snapshot / name
            expected = spec["weight_sha256"] if name == "model.safetensors" else None
            if destination.exists() and destination.stat().st_size:
                checksum = sha256(destination)
                if expected is None or checksum == expected:
                    files[name] = checksum
                    print(f"Cached: {spec['repo']}/{name}", flush=True)
                    continue
                raise RuntimeError(f"Checksum mismatch in existing {destination}; preserve it and investigate")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            failures = []
            for source in sources:
                url = f"{source}/{spec['repo']}/resolve/{spec['revision']}/{name}"
                print(f"Downloading {spec['repo']}/{name} from {source}", flush=True)
                result = subprocess.run([
                    "curl", "-fL", "--retry", "2", "--connect-timeout", "15",
                    "--speed-time", "30", "--speed-limit", "1024", "--max-time", "900",
                    "-C", "-", url, "-o", str(temporary),
                ])
                if result.returncode == 0:
                    checksum = sha256(temporary)
                    if expected is not None and checksum != expected:
                        raise RuntimeError(f"Downloaded weight checksum mismatch: {temporary}")
                    if name.endswith(".json"):
                        json.loads(temporary.read_text(encoding="utf-8"))
                    temporary.replace(destination)
                    files[name] = checksum
                    break
                failures.append(f"{source}: exit {result.returncode}")
            else:
                raise RuntimeError("Download failed: " + "; ".join(failures))
        (repo_dir / "refs").mkdir(exist_ok=True)
        (repo_dir / "refs/main").write_text(spec["revision"], encoding="utf-8")
        manifest.append({**spec, "file_sha256": files})
    manifest_path = args.cache_dir / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"All models verified. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
