"""Exercise the real QueryPlanner with scripted model output."""

import json
import unittest

from customer_service_rag.config import PlanningConfig
from customer_service_rag.planning.planner import QueryPlanner
from customer_service_rag.schemas import DialogueInput, LLMResult


class ScriptedLLM:
    def __init__(self, result):
        self.result = result
        self.payload = None

    async def complete(self, prompt, *, task, payload):
        self.payload = payload
        return LLMResult(text=json.dumps(self.result), model="scripted", is_mock=True)


class QueryPlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_resolves_followup_and_normalizes_subqueries(self):
        llm = ScriptedLLM({
            "rewritten_query": "Can a student renew student aid for the next year?",
            "intent": "renewal", "domain": "student aid",
            "sub_queries": ["student aid renewal eligibility", "student aid renewal eligibility"],
        })
        dialogue = DialogueInput(
            conversation_id="d1", current_message="Can I apply again?",
            dialogue_history=[{"role": "user", "text": "I received student aid last year."}],
        )
        plan = await QueryPlanner(llm, PlanningConfig(decompose=True)).plan(dialogue)
        self.assertEqual(plan.rewritten_query, "Can a student renew student aid for the next year?")
        self.assertEqual(plan.domain, "studentaid")
        self.assertEqual(plan.sub_queries, ["student aid renewal eligibility"])
        self.assertEqual(llm.payload["dialogue_history"][0]["role"], "user")

    async def test_hidden_history_is_not_sent_to_model(self):
        llm = ScriptedLLM({
            "rewritten_query": "What is the deadline?", "intent": "deadline",
            "domain": "unknown", "sub_queries": [],
        })
        dialogue = DialogueInput(
            conversation_id="d2", current_message="What is the deadline?",
            dialogue_history=[{"role": "user", "text": "I have student aid."}],
        )
        plan = await QueryPlanner(llm, PlanningConfig(use_history=False)).plan(dialogue)
        self.assertEqual(llm.payload["dialogue_history"], [])
        self.assertEqual(plan.dialogue_history, [])
        self.assertEqual(plan.domain, "unknown")


if __name__ == "__main__":
    unittest.main()
