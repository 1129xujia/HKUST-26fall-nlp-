"""Validation sample construction and retrieval metrics."""

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

from .schemas import Evidence, RetrievalRequest, SearchHit


class RetrievalPipeline(Protocol):
    def retrieve(self, request: RetrievalRequest) -> list[Evidence]: ...


class CandidateRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        domain: str | None = None,
        top_k: int = 20,
    ) -> list[SearchHit]: ...


@dataclass(frozen=True)
class RetrievalEvalSample:
    sample_id: str
    domain: str
    query: str
    user_gold_passage_ids: list[str]
    agent_gold_passage_ids: list[str]

    @property
    def gold_passage_ids(self) -> list[str]:
        """Return the deduplicated union used by the overall metrics and MRR."""
        return sorted(set(self.user_gold_passage_ids) | set(self.agent_gold_passage_ids))


def build_validation_samples(
    *,
    document_path: Path,
    dialogue_path: Path,
    query_mode: str = "history",
    history_turns: int = 4,
) -> list[RetrievalEvalSample]:
    document_path = Path(document_path)
    dialogue_path = Path(dialogue_path)
    if query_mode not in {"current", "history", "user_history", "current_first_user_history"}:
        raise ValueError(
            "query_mode must be 'current', 'history', 'user_history', "
            "or 'current_first_user_history'"
        )
    if type(history_turns) is not int or history_turns <= 0:
        raise ValueError("history_turns must be a positive integer")

    with document_path.open(encoding="utf-8") as stream:
        documents = json.load(stream)["doc_data"]
    with dialogue_path.open(encoding="utf-8") as stream:
        dialogues = json.load(stream)["dial_data"]

    samples: list[RetrievalEvalSample] = []
    for domain, domain_dialogues in dialogues.items():
        for dialogue in domain_dialogues:
            turns = dialogue["turns"]
            for index, turn in enumerate(turns):
                if turn.get("role") != "user":
                    continue

                user_references = list(turn.get("references", []))
                agent_references = []
                if index + 1 < len(turns) and turns[index + 1].get("role") == "agent":
                    agent_references = list(turns[index + 1].get("references", []))

                # Gold 来自 reference 的 doc_id + span，再跳回文档找到所属 section。
                def passage_ids(references: list[dict]) -> set[str]:
                    result: set[str] = set()
                    for reference in references:
                        document = documents[domain].get(reference["doc_id"])
                        if document is None:
                            continue
                        span = document["spans"].get(str(reference["id_sp"]))
                        if span is not None:
                            result.add(f"{reference['doc_id']}::{span['id_sec']}")
                    return result

                user_gold_ids = passage_ids(user_references)
                agent_gold_ids = passage_ids(agent_references)
                gold_ids = user_gold_ids | agent_gold_ids
                if not gold_ids:
                    continue

                if query_mode == "current":
                    query = turn["utterance"].strip()
                else:
                    start = max(0, index - history_turns + 1)
                    window = turns[start : index + 1]
                    if query_mode == "history":
                        query = "\n".join(
                            f"{item['role'].title()}: {item['utterance'].strip()}"
                            for item in window
                        )
                    else:
                        user_context = [
                            item["utterance"].strip()
                            for item in window
                            if item.get("role") == "user" and item["utterance"].strip()
                        ]
                        if query_mode == "current_first_user_history":
                            user_context = [user_context[-1], *user_context[:-1]]
                        query = "\n".join(user_context)

                samples.append(
                    RetrievalEvalSample(
                        sample_id=f"{dialogue['dial_id']}:{turn['turn_id']}",
                        domain=domain,
                        query=query,
                        user_gold_passage_ids=sorted(user_gold_ids),
                        agent_gold_passage_ids=sorted(agent_gold_ids),
                    )
                )
    return samples


def evaluate_pipeline(
    pipeline: RetrievalPipeline,
    samples: list[RetrievalEvalSample],
    *,
    cutoffs: tuple[int, ...] = (5, 10),
    use_domain_filter: bool = True,
    title_section_map: dict[str, str] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    if not samples:
        raise ValueError("No evaluation samples were provided")
    if not cutoffs or any(cutoff <= 0 or cutoff > 20 for cutoff in cutoffs):
        raise ValueError("cutoffs must contain values between 1 and 20")
    cutoffs = tuple(sorted(set(cutoffs)))
    max_k = max(cutoffs)
    hit_totals = {cutoff: 0 for cutoff in cutoffs}
    coverage_totals = {cutoff: 0.0 for cutoff in cutoffs}
    user_hit_totals = {cutoff: 0 for cutoff in cutoffs}
    user_coverage_totals = {cutoff: 0.0 for cutoff in cutoffs}
    agent_hit_totals = {cutoff: 0 for cutoff in cutoffs}
    agent_coverage_totals = {cutoff: 0.0 for cutoff in cutoffs}
    diagnostic_hit_totals = {cutoff: 0 for cutoff in cutoffs}
    diagnostic_coverage_totals = {cutoff: 0.0 for cutoff in cutoffs}
    diagnostic_user_hit_totals = {cutoff: 0 for cutoff in cutoffs}
    diagnostic_user_coverage_totals = {cutoff: 0.0 for cutoff in cutoffs}
    diagnostic_agent_hit_totals = {cutoff: 0 for cutoff in cutoffs}
    diagnostic_agent_coverage_totals = {cutoff: 0.0 for cutoff in cutoffs}
    user_gold_sample_count = sum(bool(sample.user_gold_passage_ids) for sample in samples)
    agent_gold_sample_count = sum(bool(sample.agent_gold_passage_ids) for sample in samples)
    reciprocal_ranks: list[float] = []
    diagnostic_reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []
    duplicate_rates: list[float] = []
    details: list[dict] = []

    for sample in samples:
        started = time.perf_counter()
        evidence = pipeline.retrieve(
            RetrievalRequest(
                query=sample.query,
                domain=sample.domain if use_domain_filter else None,
                top_k=max_k,
                trace_id=sample.sample_id,
            )
        )
        latency_ms = (time.perf_counter() - started) * 1000
        latencies_ms.append(latency_ms)

        retrieved_ids = [item.passage_id for item in evidence]
        gold = set(sample.gold_passage_ids)
        user_gold = set(sample.user_gold_passage_ids)
        agent_gold = set(sample.agent_gold_passage_ids)
        # 无前缀指标保留官方精确 ID 口径；merged_title_* 只是并行诊断口径。
        if title_section_map is not None:
            mapped_retrieved_ids = [title_section_map.get(item, item) for item in retrieved_ids]
            mapped_gold = {title_section_map.get(item, item) for item in gold}
            mapped_user_gold = {title_section_map.get(item, item) for item in user_gold}
            mapped_agent_gold = {title_section_map.get(item, item) for item in agent_gold}
        for cutoff in cutoffs:
            retrieved = set(retrieved_ids[:cutoff])
            overlap = gold & retrieved
            hit_totals[cutoff] += int(bool(overlap))
            coverage_totals[cutoff] += len(overlap) / len(gold)
            if user_gold:
                user_overlap = user_gold & retrieved
                user_hit_totals[cutoff] += int(bool(user_overlap))
                user_coverage_totals[cutoff] += len(user_overlap) / len(user_gold)
            if agent_gold:
                agent_overlap = agent_gold & retrieved
                agent_hit_totals[cutoff] += int(bool(agent_overlap))
                agent_coverage_totals[cutoff] += len(agent_overlap) / len(agent_gold)
            if title_section_map is not None:
                mapped_retrieved = set(mapped_retrieved_ids[:cutoff])
                mapped_overlap = mapped_gold & mapped_retrieved
                diagnostic_hit_totals[cutoff] += int(bool(mapped_overlap))
                diagnostic_coverage_totals[cutoff] += len(mapped_overlap) / len(mapped_gold)
                if mapped_user_gold:
                    mapped_user_overlap = mapped_user_gold & mapped_retrieved
                    diagnostic_user_hit_totals[cutoff] += int(bool(mapped_user_overlap))
                    diagnostic_user_coverage_totals[cutoff] += (
                        len(mapped_user_overlap) / len(mapped_user_gold)
                    )
                if mapped_agent_gold:
                    mapped_agent_overlap = mapped_agent_gold & mapped_retrieved
                    diagnostic_agent_hit_totals[cutoff] += int(bool(mapped_agent_overlap))
                    diagnostic_agent_coverage_totals[cutoff] += (
                        len(mapped_agent_overlap) / len(mapped_agent_gold)
                    )

        first_rank = next(
            (rank for rank, item in enumerate(retrieved_ids, start=1) if item in gold),
            None,
        )
        reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
        if title_section_map is not None:
            diagnostic_first_rank = next(
                (
                    rank
                    for rank, item in enumerate(mapped_retrieved_ids, start=1)
                    if item in mapped_gold
                ),
                None,
            )
            diagnostic_reciprocal_ranks.append(
                0.0 if diagnostic_first_rank is None else 1.0 / diagnostic_first_rank
            )
        duplicate_rates.append(
            0.0
            if not retrieved_ids
            else 1.0 - len(set(retrieved_ids)) / len(retrieved_ids)
        )
        detail = {
            **asdict(sample),
            "gold_passage_ids": sample.gold_passage_ids,
            "retrieved_passage_ids": retrieved_ids,
            "first_relevant_rank": first_rank,
            "latency_ms": latency_ms,
        }
        if title_section_map is not None:
            detail.update({
                "merged_title_gold_passage_ids": sorted(mapped_gold),
                "merged_title_retrieved_passage_ids": mapped_retrieved_ids,
                "merged_title_first_relevant_rank": diagnostic_first_rank,
            })
        details.append(detail)
        if on_progress is not None:
            on_progress(len(details), len(samples))

    sorted_latency = sorted(latencies_ms)

    def percentile(fraction: float) -> float:
        index = round((len(sorted_latency) - 1) * fraction)
        return sorted_latency[index]

    count = len(samples)
    summary = {
        "sample_count": count,
        "mrr": statistics.fmean(reciprocal_ranks),
        "mean_latency_ms": statistics.fmean(latencies_ms),
        "p50_latency_ms": percentile(0.50),
        "p95_latency_ms": percentile(0.95),
        "mean_duplicate_rate": statistics.fmean(duplicate_rates),
        "user_gold_sample_count": user_gold_sample_count,
        "agent_gold_sample_count": agent_gold_sample_count,
    }
    if title_section_map is not None:
        summary["merged_title_mrr"] = statistics.fmean(diagnostic_reciprocal_ranks)
        summary["mapped_title_section_count"] = len(title_section_map)
    for cutoff in cutoffs:
        hit_rate = hit_totals[cutoff] / count
        recall = coverage_totals[cutoff] / count
        summary[f"hit_rate@{cutoff}"] = hit_rate
        summary[f"recall@{cutoff}"] = recall
        summary[f"user_gold_hit_rate@{cutoff}"] = (
            user_hit_totals[cutoff] / user_gold_sample_count
            if user_gold_sample_count else 0.0
        )
        summary[f"user_gold_recall@{cutoff}"] = (
            user_coverage_totals[cutoff] / user_gold_sample_count
            if user_gold_sample_count else 0.0
        )
        summary[f"agent_gold_hit_rate@{cutoff}"] = (
            agent_hit_totals[cutoff] / agent_gold_sample_count
            if agent_gold_sample_count else 0.0
        )
        summary[f"agent_gold_recall@{cutoff}"] = (
            agent_coverage_totals[cutoff] / agent_gold_sample_count
            if agent_gold_sample_count else 0.0
        )
        if title_section_map is not None:
            summary[f"merged_title_hit_rate@{cutoff}"] = (
                diagnostic_hit_totals[cutoff] / count
            )
            summary[f"merged_title_recall@{cutoff}"] = (
                diagnostic_coverage_totals[cutoff] / count
            )
            summary[f"merged_title_user_gold_hit_rate@{cutoff}"] = (
                diagnostic_user_hit_totals[cutoff] / user_gold_sample_count
                if user_gold_sample_count else 0.0
            )
            summary[f"merged_title_user_gold_recall@{cutoff}"] = (
                diagnostic_user_coverage_totals[cutoff] / user_gold_sample_count
                if user_gold_sample_count else 0.0
            )
            summary[f"merged_title_agent_gold_hit_rate@{cutoff}"] = (
                diagnostic_agent_hit_totals[cutoff] / agent_gold_sample_count
                if agent_gold_sample_count else 0.0
            )
            summary[f"merged_title_agent_gold_recall@{cutoff}"] = (
                diagnostic_agent_coverage_totals[cutoff] / agent_gold_sample_count
                if agent_gold_sample_count else 0.0
            )
    return {"summary": summary, "samples": details}


def evaluate_candidate_recall(
    retriever: CandidateRetriever,
    samples: list[RetrievalEvalSample],
    *,
    cutoffs: tuple[int, ...] = (30, 50, 100, 200),
    use_domain_filter: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Measure raw single-retriever candidate recall before fusion and reranking."""
    if not samples:
        raise ValueError("No evaluation samples were provided")
    if not cutoffs or any(type(cutoff) is not int or cutoff <= 0 for cutoff in cutoffs):
        raise ValueError("cutoffs must contain positive integers")
    cutoffs = tuple(sorted(set(cutoffs)))
    max_k = max(cutoffs)
    user_count = sum(bool(sample.user_gold_passage_ids) for sample in samples)
    agent_count = sum(bool(sample.agent_gold_passage_ids) for sample in samples)
    totals = {
        group: {
            "hit": {cutoff: 0 for cutoff in cutoffs},
            "recall": {cutoff: 0.0 for cutoff in cutoffs},
        }
        for group in ("all", "user", "agent")
    }
    details: list[dict] = []

    for sample in samples:
        hits = retriever.search(
            sample.query,
            domain=sample.domain if use_domain_filter else None,
            top_k=max_k,
        )
        retrieved_ids = [hit.passage.passage_id for hit in hits]
        rank_by_id = {passage_id: rank for rank, passage_id in enumerate(retrieved_ids, 1)}
        groups = {
            "all": set(sample.gold_passage_ids),
            "user": set(sample.user_gold_passage_ids),
            "agent": set(sample.agent_gold_passage_ids),
        }
        for cutoff in cutoffs:
            retrieved = set(retrieved_ids[:cutoff])
            for group, gold in groups.items():
                if not gold:
                    continue
                overlap = gold & retrieved
                totals[group]["hit"][cutoff] += int(bool(overlap))
                totals[group]["recall"][cutoff] += len(overlap) / len(gold)

        details.append({
            "sample_id": sample.sample_id,
            "domain": sample.domain,
            "user_gold_passage_ids": sample.user_gold_passage_ids,
            "agent_gold_passage_ids": sample.agent_gold_passage_ids,
            "gold_ranks": {
                passage_id: rank_by_id.get(passage_id)
                for passage_id in sample.gold_passage_ids
            },
            "retrieved_count": len(retrieved_ids),
        })
        if on_progress is not None:
            on_progress(len(details), len(samples))

    counts = {"all": len(samples), "user": user_count, "agent": agent_count}
    summary = {
        "sample_count": len(samples),
        "user_gold_sample_count": user_count,
        "agent_gold_sample_count": agent_count,
    }
    for cutoff in cutoffs:
        for group in ("all", "user", "agent"):
            prefix = "" if group == "all" else f"{group}_gold_"
            denominator = counts[group]
            summary[f"{prefix}hit_rate@{cutoff}"] = (
                totals[group]["hit"][cutoff] / denominator if denominator else 0.0
            )
            summary[f"{prefix}recall@{cutoff}"] = (
                totals[group]["recall"][cutoff] / denominator if denominator else 0.0
            )
        summary[f"no_hit_rate@{cutoff}"] = 1.0 - summary[f"hit_rate@{cutoff}"]
    return {"summary": summary, "samples": details}
