from __future__ import annotations

from time import perf_counter

from customer_service_rag.config import BudgetConfig
from customer_service_rag.schemas import TokenUsage


class BudgetExceeded(RuntimeError):
    pass


class BudgetTracker:
    def __init__(self, config: BudgetConfig) -> None:
        self.config = config
        self.started = perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (perf_counter() - self.started) * 1000

    def check(self, usages: list[TokenUsage], reserved_tokens: int = 0) -> None:
        if reserved_tokens < 0:
            raise ValueError("reserved_tokens must be nonnegative")
        if self.elapsed_ms >= self.config.max_latency_seconds * 1000:
            raise BudgetExceeded("latency_budget")
        totals = [usage.total_tokens for usage in usages]
        if any(total is None for total in totals):
            raise BudgetExceeded("token_usage_unavailable")
        if sum(totals) + reserved_tokens > self.config.max_total_tokens:
            raise BudgetExceeded("token_budget")
