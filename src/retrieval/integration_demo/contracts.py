"""Narrow interfaces that keep the demo parts replaceable."""

from typing import Protocol

from retrieval.schemas import Evidence, RetrievalRequest

from .part1.planner import QueryPlan
from .part3.generator import CitedAnswer


class RetrievalService(Protocol):
    def retrieve(self, request: RetrievalRequest) -> list[Evidence]: ...


class QueryPlanner(Protocol):
    def plan(self, question: str, history: list[dict[str, str]] | None = None) -> QueryPlan: ...


class ResponseGenerator(Protocol):
    def generate(self, question: str, evidence: list[Evidence]) -> CitedAnswer: ...


class QualityEvaluator(Protocol):
    def evaluate(self, answer: CitedAnswer, evidence: list[Evidence]) -> dict: ...
