"""Data contracts and metrics shared by training, evaluation and reporting."""
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
PREFIX = "Represent this sentence for searching relevant passages: "
SEED = 20261108


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def norm(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def samples_from_dialogues(path, passages):
    """Use only past/current turns as input; next-agent spans are offline labels."""
    span_map = {}
    for p in passages:
        for sid in p.span_ids:
            key = (p.doc_id, sid)
            if key in span_map:
                raise ValueError(f"Span belongs to multiple passages: {key}")
            span_map[key] = p.passage_id
    samples, skipped = [], defaultdict(int)
    for domain, dialogues in read_json(path)["dial_data"].items():
        for dialogue in dialogues:
            turns = dialogue["turns"]
            for i, turn in enumerate(turns[:-1]):
                target = turns[i + 1]
                if turn["role"] != "user" or target["role"] != "agent":
                    continue
                if target.get("da") == "respond_no_solution":
                    skipped["no_solution"] += 1
                    continue
                refs = target.get("references", [])
                keys = {(r["doc_id"], str(r["id_sp"])) for r in refs}
                if not keys or not keys.issubset(span_map):
                    skipped["empty_or_unmapped_gold"] += 1
                    continue
                query = "\n".join(f"{t['role'].title()}: {' '.join(t['utterance'].split())}"
                                  for t in turns[max(0, i - 3):i + 1])
                samples.append({"sample_id": f"{dialogue['dial_id']}:{turn['turn_id']}",
                    "dialogue_id": dialogue["dial_id"], "domain": domain,
                    "query": query, "current": turn["utterance"],
                    "gold": sorted({span_map[k] for k in keys}),
                    "evidence_spans": sorted([list(k) for k in keys])})
    return samples, dict(skipped)


def balanced_sample(samples, limit, seed=SEED):
    groups = defaultdict(list)
    for s in samples:
        groups[s["domain"]].append(s)
    rng = random.Random(seed)
    for domain in sorted(groups):
        rng.shuffle(groups[domain])
    ordered = [groups[d][i] for i in range(max(map(len, groups.values())))
               for d in sorted(groups) if i < len(groups[d])]
    return ordered[:limit] if limit else ordered


def metrics(ranking, gold):
    gold = set(gold)
    if not gold:
        raise ValueError("Empty Gold is not an evaluable query")
    top = ranking[:10]
    if len(set(ranking)) != len(ranking):
        raise ValueError("Duplicate passage IDs in ranking")
    rel = [int(p in gold) for p in top]
    first = next((i for i, r in enumerate(rel, 1) if r), None)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(gold))))
    return {"hit@10": float(any(rel)), "recall@10": sum(rel) / len(gold),
            "mrr@10": 1 / first if first else 0.0,
            "ndcg@10": sum(r / math.log2(i + 2) for i, r in enumerate(rel)) / ideal,
            "hit@50": float(bool(set(ranking[:50]) & gold)),
            "recall@50": len(set(ranking[:50]) & gold) / len(gold)}


def summarize(rows):
    import numpy as np
    keys = list(rows[0]["metrics"])
    result = {k: float(np.mean([r["metrics"][k] for r in rows])) for k in keys}
    times = [r["latency_ms"] for r in rows]
    result.update(n=len(rows), latency_mean_ms=float(np.mean(times)),
                  latency_p50_ms=float(np.percentile(times, 50)),
                  latency_p95_ms=float(np.percentile(times, 95)))
    return result
