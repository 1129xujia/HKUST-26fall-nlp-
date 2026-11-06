"""DeepSeek adapter tests with a fake HTTP transport; no network or real key."""

import json
import os
import unittest
from unittest.mock import patch

from customer_service_rag.config import LLMConfig
from customer_service_rag.models.deepseek_llm import DeepSeekLLM


class DeepSeekAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_json_request_and_usage_without_exposing_key(self):
        seen = {}

        def transport(body, key, timeout):
            seen["body"] = json.loads(body)
            seen["key"] = key
            seen["timeout"] = timeout
            return {
                "choices": [{"finish_reason": "stop", "message": {"content": '{"x": 1}'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            }

        llm = DeepSeekLLM(
            LLMConfig(backend="deepseek", model="deepseek-flash"), transport=transport,
        )
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only-placeholder"}):
            result = await llm.complete("Return JSON", task="plan", payload={})
        self.assertEqual(seen["body"]["response_format"], {"type": "json_object"})
        self.assertEqual(seen["body"]["model"], "deepseek-flash")
        self.assertEqual(seen["key"], "test-only-placeholder")
        self.assertEqual(json.loads(result.text), {"x": 1})
        self.assertEqual(result.usage.total_tokens, 16)
        self.assertNotIn("test-only-placeholder", repr(llm.calls))

    async def test_missing_key_fails_before_transport(self):
        llm = DeepSeekLLM(
            LLMConfig(backend="deepseek", model="deepseek-flash"),
            transport=lambda *_: self.fail("transport should not be called"),
        )
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
                await llm.complete("Return JSON", task="generate", payload={})

    async def test_truncated_response_is_rejected(self):
        llm = DeepSeekLLM(
            LLMConfig(backend="deepseek", model="deepseek-flash"),
            transport=lambda *_: {
                "choices": [{"finish_reason": "length", "message": {"content": '{"x":'}}],
            },
        )
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only-placeholder"}):
            with self.assertRaisesRegex(RuntimeError, "complete JSON"):
                await llm.complete("Return JSON", task="generate", payload={})


if __name__ == "__main__":
    unittest.main()
