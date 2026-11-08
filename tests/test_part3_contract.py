"""Part 3 boundary checks using Part 2's public evidence shape."""

import json
import unittest

from customer_service_rag.part3 import EvidenceResponseGenerator, run_once
from customer_service_rag.schemas import DialogueInput, LLMResult, QueryPlan
from retrieval import Evidence
from retrieval.graph_node import make_retrieval_node


class FixedPlanner:
    async def plan(self, dialogue):
        return QueryPlan(
            **dialogue.model_dump(),
            rewritten_query="How can a student renew financial aid?",
            intent="renewal", domain="studentaid",
            sub_queries=["student aid renewal documents"],
        )


class FakeLLM:
    async def complete(self, prompt, *, task, payload):
        first = payload["passages"][0]
        return LLMResult(
            text=json.dumps({"answer": first["text"],
                             "citations": ["E1"], "status": "answered"}),
            model="test", is_mock=True,
        )


class BadCitationLLM(FakeLLM):
    async def complete(self, prompt, *, task, payload):
        return LLMResult(
            text=json.dumps({"answer": "Unsupported", "citations": ["other:p9"],
                             "status": "answered"}),
            model="test", is_mock=True,
        )


class TwoEvidenceLLM(FakeLLM):
    async def complete(self, prompt, *, task, payload):
        return LLMResult(
            text=json.dumps({"answer": "Submit both documents before 15 May.",
                             "citations": ["E1", "E2"], "status": "answered"}),
            model="test", is_mock=True,
        )


class FixedPipeline:
    class Config:
        final_top_k = 6

    config = Config()

    def __init__(self, items):
        self.items = items
        self.last_request = None

    def retrieve(self, request):
        self.last_request = request
        return self.items


def evidence(passage_id, text):
    return Evidence(
        passage_id=passage_id, doc_id="aid-1", domain="studentaid",
        title="Student Aid", section_title="Renewal", text=text,
        span_ids=[], source_retrievers=["bm25"], matched_query="renewal",
    )


class Part3ContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_planning_retrieval_generation_preserves_order_and_citation(self):
        seen = {}
        first = evidence("p1", "Apply for renewal through the aid portal.")
        second = evidence("p2", "Bring your supporting documents.")

        pipeline = FixedPipeline([first, second])
        real_node = make_retrieval_node(pipeline)

        def node(state):
            seen.update(state)
            return real_node(state)

        dialogue = DialogueInput(
            conversation_id="d1", current_message="Can I apply again?",
            dialogue_history=[{"role": "user", "text": "I had student aid last year."}],
        )
        result = await run_once(
            dialogue, planner=FixedPlanner(), retrieval_node=node,
            generator=EvidenceResponseGenerator(FakeLLM()), top_k=2,
        )
        self.assertEqual(seen["query"], "How can a student renew financial aid?")
        self.assertEqual(seen["sub_queries"], ["student aid renewal documents"])
        self.assertEqual(seen["domain"], "studentaid")
        self.assertEqual(seen["retrieval_top_k"], 2)
        self.assertEqual(pipeline.last_request, result.request)
        self.assertEqual([item.passage_id for item in result.evidence], ["p1", "p2"])
        self.assertEqual(result.response.citations, ["p1"])

    async def test_empty_evidence_returns_insufficient(self):
        dialogue = DialogueInput(conversation_id="d2", current_message="What next?")
        result = await run_once(
            dialogue, planner=FixedPlanner(), retrieval_node=lambda _: {"evidence": []},
            generator=EvidenceResponseGenerator(FakeLLM()),
        )
        self.assertEqual(result.response.status.value, "insufficient_evidence")
        self.assertEqual(result.response.citations, [])

    async def test_rejects_citation_outside_evidence(self):
        dialogue = DialogueInput(conversation_id="d3", current_message="Can I renew?")
        with self.assertRaisesRegex(ValueError, "not in the supplied evidence"):
            await run_once(
                dialogue, planner=FixedPlanner(),
                retrieval_node=lambda _: {"evidence": [evidence("p1", "Renew online.").to_dict()]},
                generator=EvidenceResponseGenerator(BadCitationLLM()),
            )

    async def test_two_evidence_answer_keeps_both_citations(self):
        dialogue = DialogueInput(conversation_id="d4", current_message="What and when?")
        result = await run_once(
            dialogue, planner=FixedPlanner(),
            retrieval_node=lambda _: {"evidence": [
                evidence("documents", "Submit both documents.").to_dict(),
                evidence("deadline", "The deadline is 15 May.").to_dict(),
            ]},
            generator=EvidenceResponseGenerator(TwoEvidenceLLM()),
        )
        self.assertEqual(result.response.citations, ["documents", "deadline"])

    async def test_part2_section_id_is_the_final_citation(self):
        dialogue = DialogueInput(conversation_id="d5", current_message="Can I renew?")
        item = evidence("aid-1::section-2", "Renew online.")
        result = await run_once(
            dialogue, planner=FixedPlanner(),
            retrieval_node=lambda _: {"evidence": [item.to_dict()]},
            generator=EvidenceResponseGenerator(FakeLLM()),
        )
        self.assertEqual(result.response.citations, ["aid-1::section-2"])


if __name__ == "__main__":
    unittest.main()
