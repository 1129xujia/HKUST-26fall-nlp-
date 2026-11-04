"""Controlled BM25 tokenization and field-weight ablations."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import re

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from experiments.optimize_retrieval import balanced_samples, run_configuration
from retrieval.config import RetrievalConfig
from retrieval.corpus import load_passages_jsonl
from retrieval.dense import DenseRetriever
from retrieval.evaluation import build_validation_samples
from retrieval.schemas import SearchHit


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
ROLE_TOKENS = {"user", "agent"}


def words(text):
    return TOKEN_PATTERN.findall(text.lower())


def add_bigrams(tokens):
    return tokens + [f"{left}__{right}" for left, right in zip(tokens, tokens[1:])]


class ExperimentalBM25:
    def __init__(self, passages, *, remove_stopwords=False, bigrams=False, title_boost=0, strip_roles=False):
        self.passages = passages
        self.remove_stopwords = remove_stopwords
        self.bigrams = bigrams
        self.strip_roles = strip_roles
        corpus = []
        for passage in passages:
            tokens = self._tokens(passage.index_text, query=False)
            title_text = " ".join([passage.title, *passage.parent_titles, passage.section_title])
            tokens.extend(words(title_text) * title_boost)
            corpus.append(tokens)
        self.index = BM25Okapi(corpus)

    def _tokens(self, text, *, query):
        tokens = words(text)
        if query and self.strip_roles:
            tokens = [token for token in tokens if token not in ROLE_TOKENS]
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS]
        return add_bigrams(tokens) if self.bigrams else tokens

    def search(self, query, *, domain=None, top_k=20):
        tokens = self._tokens(query, query=True)
        if not tokens or top_k <= 0:
            return []
        scores = self.index.get_scores(tokens)
        hits = []
        for row in sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True):
            passage = self.passages[row]
            if domain and passage.domain != domain:
                continue
            if not any(token in self.index.doc_freqs[row] for token in tokens):
                continue
            hits.append(SearchHit(passage, "bm25", len(hits) + 1, float(scores[row]), query))
            if len(hits) >= top_k:
                break
        return hits


def candidate_summary(rows, key, cutoffs=(10, 50, 200)):
    result = {}
    for cutoff in cutoffs:
        hits = 0
        recall = 0.0
        for row in rows:
            gold = set(row["sample"].gold_passage_ids)
            retrieved = {hit.passage.passage_id for hit in row[key][:cutoff]}
            overlap = gold & retrieved
            hits += int(bool(overlap))
            recall += len(overlap) / len(gold)
        result[f"hit_rate@{cutoff}"] = hits / len(rows)
        result[f"recall@{cutoff}"] = recall / len(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("experiments/output/bm25_ablation.json"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(Path(".model_cache").resolve()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    passages = load_passages_jsonl(args.artifact_dir / "passages.jsonl")
    samples = balanced_samples(build_validation_samples(
        document_path=Path("multidoc2dial/multidoc2dial_doc.json"),
        dialogue_path=Path("multidoc2dial/multidoc2dial_dial_validation.json"),
        query_mode="history", history_turns=4,
    ), args.limit)
    base_config = RetrievalConfig.for_mode("rrf", artifact_dir=args.artifact_dir)
    dense = DenseRetriever.from_artifacts(
        args.artifact_dir, base_config.embedding_model, passages=passages,
        query_prefix=base_config.dense_query_prefix,
        passage_prefix=base_config.dense_passage_prefix,
        device=args.device,
    )
    rows = []
    print("Caching Dense Top 100 once...", flush=True)
    for index, sample in enumerate(samples, 1):
        rows.append({
            "sample": sample,
            "dense": dense.search(sample.query, domain=sample.domain, top_k=100),
        })
        if index % 10 == 0:
            print(f"  dense {index}/{len(samples)}", flush=True)

    variants = {
        "baseline": {},
        "strip_roles": {"strip_roles": True},
        "remove_stopwords": {"remove_stopwords": True},
        "bigrams": {"bigrams": True},
        "title_boost_1": {"title_boost": 1},
        "title_boost_2": {"title_boost": 2},
        "title_boost_2_bigrams_strip_roles": {
            "title_boost": 2, "bigrams": True, "strip_roles": True,
        },
    }
    results = {}
    for name, options in variants.items():
        print(f"Building/evaluating BM25 variant: {name}", flush=True)
        retriever = ExperimentalBM25(passages, **options)
        variant_rows = []
        for row in rows:
            sample = row["sample"]
            variant_rows.append({
                **row,
                "bm25": retriever.search(sample.query, domain=sample.domain, top_k=200),
            })
        hybrid = run_configuration(
            variant_rows,
            dense_k=100, bm25_k=50, rrf_k=30,
            fused_k=400, final_k=10, selector=None,
        )
        results[name] = {
            "options": options,
            "bm25_candidate": candidate_summary(variant_rows, "bm25"),
            "hybrid": hybrid,
        }

    ranked = sorted(
        results,
        key=lambda name: (
            results[name]["hybrid"]["recall@10"],
            results[name]["hybrid"]["hit_rate@10"],
            results[name]["hybrid"]["mrr@10"],
        ),
        reverse=True,
    )
    output = {
        "configuration": {
            "sample_count": len(samples), "query_mode": "history",
            "dense_model": base_config.embedding_model,
            "dense_k": 100, "bm25_k": 50, "rrf_k": 30,
            "reranker": False, "deduplication": False,
        },
        "results": results,
        "ranking": ranked,
        "best": {"name": ranked[0], **results[ranked[0]]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(output["best"], indent=2), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
