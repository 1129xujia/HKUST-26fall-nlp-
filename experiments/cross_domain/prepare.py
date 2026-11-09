"""Prepare shared project assets for the cross-domain experiment."""

import argparse

from experiments import ROOT
from customer_service_rag.assets import download_dataset, download_models
from customer_service_rag.config import load_config
from customer_service_rag.data.official import prepare_official
from customer_service_rag.orchestration.retrieval import build_index_for_config


def model_config():
    return load_config(ROOT / "configs/deepseek.yaml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    args = parser.parse_args()
    if args.download_data:
        download_dataset(ROOT)
    if args.download_models:
        download_models(ROOT, endpoint=args.endpoint)
    config = model_config()
    processed = prepare_official(config)
    manifest = build_index_for_config(config)
    print(f"Prepared shared corpus: {processed['passages']} passages; index: {manifest}")


if __name__ == "__main__":
    main()
