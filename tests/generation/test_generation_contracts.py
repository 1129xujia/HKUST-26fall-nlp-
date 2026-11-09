"""Generation boundary checks using the retrieval package's public contracts."""

import json
import unittest

from customer_service_rag.generation.generator import EvidenceResponseGenerator
from customer_service_rag.schemas import DialogueInput, LLMResult, QueryPlan
from customer_service_rag.retrieval import Evidence, RetrievalRequest


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


class GenerationContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_planning_retrieval_generation_preserves_order_and_citation(self):
        first = evidence("p1", "Apply for renewal through the aid portal.")
        second = evidence("p2", "Bring your supporting documents.")

        pipeline = FixedPipeline([first, second])
        dialogue = DialogueInput(
            conversation_id="d1", current_message="Can I apply again?",
            dialogue_history=[{"role": "user", "text": "I had student aid last year."}],
        )
        plan = await FixedPlanner().plan(dialogue)
        request = RetrievalRequest(
            query=plan.rewritten_query, sub_queries=plan.sub_queries,
            domain=plan.domain, top_k=2, trace_id=plan.conversation_id,
        )
        retrieved = pipeline.retrieve(request)
        response = await EvidenceResponseGenerator(FakeLLM()).generate(plan, retrieved)
        self.assertEqual(pipeline.last_request.query, "How can a student renew financial aid?")
        self.assertEqual(pipeline.last_request.sub_queries, ["student aid renewal documents"])
        self.assertEqual(pipeline.last_request.domain, "studentaid")
        self.assertEqual(pipeline.last_request.top_k, 2)
        self.assertEqual(pipeline.last_request, request)
        self.assertEqual([item.passage_id for item in retrieved], ["p1", "p2"])
        self.assertEqual(response.citations, ["p1"])

    async def test_empty_evidence_returns_insufficient(self):
        dialogue = DialogueInput(conversation_id="d2", current_message="What next?")
        plan = await FixedPlanner().plan(dialogue)
        response = await EvidenceResponseGenerator(FakeLLM()).generate(plan, [])
        self.assertEqual(response.status.value, "insufficient_evidence")
        self.assertEqual(response.citations, [])

    async def test_rejects_citation_outside_evidence(self):
        dialogue = DialogueInput(conversation_id="d3", current_message="Can I renew?")
        plan = await FixedPlanner().plan(dialogue)
        with self.assertRaisesRegex(ValueError, "not in the supplied evidence"):
            await EvidenceResponseGenerator(BadCitationLLM()).generate(
                plan, [evidence("p1", "Renew online.")],
            )

    async def test_two_evidence_answer_keeps_both_citations(self):
        dialogue = DialogueInput(conversation_id="d4", current_message="What and when?")
        plan = await FixedPlanner().plan(dialogue)
        response = await EvidenceResponseGenerator(TwoEvidenceLLM()).generate(
            plan, [evidence("documents", "Submit both documents."),
                   evidence("deadline", "The deadline is 15 May.")],
        )
        self.assertEqual(response.citations, ["documents", "deadline"])

    async def test_part2_section_id_is_the_final_citation(self):
        dialogue = DialogueInput(conversation_id="d5", current_message="Can I renew?")
        item = evidence("aid-1::section-2", "Renew online.")
        plan = await FixedPlanner().plan(dialogue)
        response = await EvidenceResponseGenerator(FakeLLM()).generate(plan, [item])
        self.assertEqual(response.citations, ["aid-1::section-2"])


if __name__ == "__main__":
    unittest.main()
