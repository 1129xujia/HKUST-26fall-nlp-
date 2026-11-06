"""DeepSeek protocol tests using an injected local transport, no real requests."""

import asyncio
import json
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from optimaization_v2 import (
    DeepSeekLLM, LLMQueryRewriter, LLMRevisionGenerator, MainEvaluatorAdapter,
    create_deepseek_controller,
)
from optimaization_v2.budget import BudgetExceeded, BudgetLedger
from optimaization_v2.schemas import BudgetState, Limits


class Transport:
    DEFAULT_CONFIG_PATH = Path("offline-test-config.json")
    DEFAULT_MODEL = "test-deepseek-model"
    API_URL = "https://api.deepseek.com/chat/completions"

    def __init__(self):
        self.requests = []
        self.response = {
            "model": "test-deepseek-model",
            "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }
        self.failure = None

    def load_local_config(self, path):
        return {"api_key": "offline-test-credential", "model": "configured-test-model"}

    def call_deepseek(self, payload, api_key):
        self.requests.append((payload, api_key))
        if self.failure:
            raise RuntimeError(self.failure)
        return self.response, 0.01


class DeepSeekConfigurationTests(TestCase):
    def test_reuses_config_without_sending_request(self):
        transport = Transport()
        client = DeepSeekLLM(transport=transport)
        self.assertEqual(client.model_name, "configured-test-model")
        self.assertFalse(transport.requests)
        self.assertNotIn("offline-test-credential", repr(client))

    def test_request_matches_existing_json_protocol(self):
        client = DeepSeekLLM(transport=Transport(), max_output_tokens=700)
        payload = client.build_payload("Return JSON", task="plan")
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["max_tokens"], 700)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["temperature"], 0)
        self.assertNotIn("offline-test-credential", json.dumps(payload))

    def test_default_adapters_resolve_deepseek(self):
        client = DeepSeekLLM(transport=Transport())
        with patch("optimaization_v2.deepseek.DeepSeekLLM", return_value=client):
            for adapter in (LLMRevisionGenerator(), LLMQueryRewriter(), MainEvaluatorAdapter()):
                self.assertIs(adapter.llm, client)
                self.assertEqual(adapter.max_output_tokens, client.max_output_tokens)

    def test_factory_shares_one_client_across_all_model_components(self):
        client = DeepSeekLLM(transport=Transport(), max_output_tokens=768)
        controller = create_deepseek_controller(retrieval=object(), llm=client)
        for component in (controller.generator, controller.rewriter, controller.evaluator):
            self.assertIs(component.llm, client)
            self.assertEqual(component.max_output_tokens, 768)

    def test_mismatched_output_reservation_is_rejected(self):
        client = DeepSeekLLM(transport=Transport(), max_output_tokens=1024)
        with self.assertRaises(ValueError):
            LLMRevisionGenerator(client, max_output_tokens=384)

    def test_missing_key_is_explicit(self):
        transport = Transport()
        transport.load_local_config = lambda path: {}
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}), self.assertRaises(ValueError):
            DeepSeekLLM(transport=transport)


class DeepSeekCompletionTests(IsolatedAsyncioTestCase):
    async def test_provider_usage_is_reported_for_all_tasks(self):
        transport = Transport()
        client = DeepSeekLLM(transport=transport)
        ledger = BudgetLedger(Limits(), BudgetState(used_tokens=10))
        for task in ("plan", "generate", "evaluate"):
            result = await ledger.complete(client, "Return JSON", task=task, payload={})
            self.assertFalse(result.is_mock)
            self.assertEqual(result.usage.source, "reported")
            self.assertEqual(json.loads(result.text), {"ok": True})
        self.assertEqual(ledger.used_tokens, 58)
        self.assertEqual(len(transport.requests), 3)
        self.assertEqual([c["task"] for c in client.calls], ["plan", "generate", "evaluate"])

    async def test_payload_is_not_uploaded_separately(self):
        transport = Transport()
        client = DeepSeekLLM(transport=transport)
        await client.complete("Only this prompt", task="plan", payload={"gold": "private-gold"})
        self.assertNotIn("private-gold", json.dumps(transport.requests[0][0]))

    async def test_truncated_completion_is_rejected_without_retry(self):
        transport = Transport()
        transport.response["choices"][0]["finish_reason"] = "length"
        client = DeepSeekLLM(transport=transport)
        with self.assertRaisesRegex(ValueError, "normally"):
            await client.complete("x", task="plan", payload={})
        self.assertEqual(len(transport.requests), 1)
        self.assertEqual(client.usage[0].total_tokens, 16)

    async def test_non_object_or_invalid_json_is_rejected(self):
        for text in ("[]", "not json", "{} trailing text"):
            transport = Transport()
            transport.response["choices"][0]["message"]["content"] = text
            client = DeepSeekLLM(transport=transport)
            with self.subTest(text=text), self.assertRaises(ValueError):
                await client.complete("x", task="generate", payload={})

    async def test_missing_usage_is_not_charged_as_zero(self):
        transport = Transport()
        transport.response.pop("usage")
        client = DeepSeekLLM(transport=transport)
        ledger = BudgetLedger(Limits(), BudgetState())
        with self.assertRaisesRegex(BudgetExceeded, "token_usage_unavailable"):
            await ledger.complete(client, "x", task="evaluate", payload={})
        self.assertIsNone(ledger.used_tokens)

    async def test_error_is_redacted_and_never_retried(self):
        transport = Transport()
        transport.failure = "bad request offline-test-credential"
        client = DeepSeekLLM(transport=transport)
        with self.assertRaises(RuntimeError) as raised:
            await client.complete("x", task="generate", payload={})
        self.assertNotIn("offline-test-credential", str(raised.exception))
        self.assertNotIn("offline-test-credential", json.dumps(client.calls))
        self.assertEqual(len(transport.requests), 1)

    async def test_cancellation_records_unknown_usage(self):
        transport = Transport()
        client = DeepSeekLLM(transport=transport)

        async def pending(*args):
            await asyncio.sleep(1)

        with patch("optimaization_v2.deepseek.asyncio.to_thread", side_effect=pending):
            task = asyncio.create_task(client.complete("x", task="generate", payload={}))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(client.calls[-1]["status"], "cancelled")
        self.assertIsNone(client.usage[-1].total_tokens)
