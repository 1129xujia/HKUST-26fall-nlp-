"""One deadline and one token ledger for all optimization branches."""

from __future__ import annotations

import asyncio
from time import monotonic

from .schemas import BudgetState, Limits


class BudgetExceeded(RuntimeError):
    pass


class BudgetLedger:
    def __init__(self, limits: Limits, initial: BudgetState):
        self.limits = limits
        self.used_tokens = initial.used_tokens
        self.previous_elapsed = initial.elapsed_seconds
        self.started = monotonic()

    @property
    def elapsed_seconds(self):
        return self.previous_elapsed + monotonic() - self.started

    def check(self, reserved_tokens: int = 0):
        if self.elapsed_seconds >= self.limits.max_latency_seconds:
            raise BudgetExceeded("latency_budget")
        if self.used_tokens is None:
            raise BudgetExceeded("token_usage_unavailable")
        if self.used_tokens + reserved_tokens > self.limits.max_total_tokens:
            raise BudgetExceeded("token_budget")

    async def run(self, factory):
        """Use a factory so no coroutine is leaked when a budget check fails."""
        self.check()
        remaining = self.limits.max_latency_seconds - self.elapsed_seconds
        try:
            result = await asyncio.wait_for(factory(), timeout=remaining)
        except asyncio.TimeoutError as exc:
            raise BudgetExceeded("latency_budget") from exc
        self.check()
        return result

    async def complete(self, llm, prompt: str, *, task: str, payload: dict,
                       max_output_tokens: int = 384):
        # Conservative reservation, not a tokenizer measurement. Exact provider
        # counts are charged afterwards. No silent context truncation.
        maximum = getattr(llm, "max_output_tokens", max_output_tokens)
        reservation = len(prompt.encode("utf-8")) + 128 + max(max_output_tokens, maximum)
        self.check(reservation)
        try:
            result = await llm.complete(prompt, task=task, payload=payload)
        except (Exception, asyncio.CancelledError):
            # The provider may already have consumed tokens. Never report zero
            # when execution failed/cancelled without a complete usage record.
            self.used_tokens = None
            raise
        self.record_usage(result.usage)
        return result

    def record_usage(self, usage):
        """For injected non-LLM services that supply a TokenUsage record."""
        total = usage.total_tokens
        if total is None:
            self.used_tokens = None
        elif self.used_tokens is not None:
            self.used_tokens += total
        self.check()


class BudgetedLLM:
    """Lets the unchanged main EvaluatorAgent use this request's ledger."""

    def __init__(self, llm, budget: BudgetLedger, max_output_tokens: int = 384):
        self.llm = llm
        self.budget = budget
        self.max_output_tokens = max_output_tokens

    async def complete(self, prompt, *, task, payload):
        return await self.budget.complete(
            self.llm, prompt, task=task, payload=payload,
            max_output_tokens=self.max_output_tokens,
        )
