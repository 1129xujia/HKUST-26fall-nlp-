"""Diagnostic behavior check on synthetic mini cases; not a model-quality benchmark."""

from __future__ import annotations

import asyncio
import argparse
import json
from pathlib import Path

from customer_service_rag.config import PlanningConfig
from customer_service_rag.models.llm import MockLLM
from customer_service_rag.part3 import EvidenceResponseGenerator
from customer_service_rag.planning.planner import QueryPlanner
from customer_service_rag.schemas import DialogueInput
from retrieval_v2.retrieval import Evidence


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


async def evaluate(llm=None, limit: int | None = None) -> dict:
    samples = {item["sample_id"]: item for item in read_jsonl(ROOT / "data/samples/dialogues.jsonl")}
    passages = {f'{item["doc_id"]}:{item["passage_id"]}': item
                for item in read_jsonl(ROOT / "data/samples/passages.jsonl")}
    cases = json.loads((ROOT / "tests/fixtures/part3_behavior_cases.json").read_text(encoding="utf-8"))
    if limit is not None:
        cases = cases[:limit]
    llm = llm or MockLLM()
    planner = QueryPlanner(llm, PlanningConfig())
    generator = EvidenceResponseGenerator(llm)
    rows = []
    for case in cases:
        sample = samples.get(case["id"])
        dialogue = DialogueInput.model_validate(case.get("dialogue") or sample["dialogue"])
        plan = await planner.plan(dialogue)
        query = plan.rewritten_query.casefold()
        query_checks = {
            "expected_terms": all(term.casefold() in query for term in case["query_must_contain"]),
            "standalone_format": "conversation context:" not in query and "current question:" not in query,
            "domain": plan.domain == case["expected_domain"],
        }
        row = {"id": case["id"], "rewritten_query": plan.rewritten_query,
               "domain": plan.domain, "query_checks": query_checks}
        if sample and sample["gold_citations"]:
            evidence = [Evidence(
                passage_id=passages[key]["passage_id"], doc_id=passages[key]["doc_id"],
                domain=passages[key]["domain"], title=passages[key]["title"],
                section_title="", text=passages[key]["text"], span_ids=[],
                source_retrievers=["gold_fixture"], matched_query=plan.rewritten_query,
            ) for key in sample["gold_citations"]]
            response = await generator.generate(plan, evidence)
            answer = response.answer.casefold()
            row["response"] = {
                "answer": response.answer, "citations": response.citations,
                "checks": {
                    "required_terms": all(term.casefold() in answer for term in case["answer_must_contain"]),
                    "required_citations": {
                        passages[key]["passage_id"] for key in sample["gold_citations"]
                    }.issubset(response.citations),
                },
            }
        rows.append(row)
    checks = [value for row in rows for value in row["query_checks"].values()]
    checks += [value for row in rows if "response" in row
               for value in row["response"]["checks"].values()]
    return {"model": llm.model_name, "source": "synthetic mini + 2 supplemental cases",
            "note": "Gold evidence is injected for generation; this does not test retrieval or semantic grounding.",
            "passed_checks": sum(checks), "total_checks": len(checks), "cases": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["mock", "deepseek"], default="mock")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    llm = None
    if args.backend == "deepseek":
        from customer_service_rag.config import LLMConfig
        from customer_service_rag.models.llm import build_llm
        llm = build_llm(LLMConfig(backend="deepseek", model="deepseek-flash",
                                  max_new_tokens=512))
    report = asyncio.run(evaluate(llm, args.limit))
    output = ROOT / f"reports/part3_behavior_{args.backend}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['passed_checks']}/{report['total_checks']} checks passed; report: {output}")
