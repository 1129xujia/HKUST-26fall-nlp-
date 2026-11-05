"""Request-scoped evaluation/optimization loop with bounded execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from uuid import uuid4

from customer_service_rag.config import AppConfig
from customer_service_rag.generation.citations import validate_citations
from customer_service_rag.generation.generator import ResponseGenerator
from customer_service_rag.interfaces import Evaluator, Generator, LLMClient, Optimizer, Retriever
from customer_service_rag.models.llm import build_llm
from customer_service_rag.optimization.optimizer import OptimizerAgent
from customer_service_rag.orchestration.actions import apply_action
from customer_service_rag.orchestration.budget import BudgetExceeded, BudgetTracker
from customer_service_rag.orchestration.escalation import package_escalation
from customer_service_rag.orchestration.state import record_event, transition
from customer_service_rag.planning.planner import QueryPlanner
from customer_service_rag.quality.evaluator import EvaluatorAgent
from customer_service_rag.quality.policy import DecisionPolicy
from customer_service_rag.schemas import (
    Decision,
    DialogueInput,
    EscalationPackage,
    EvaluationFeedback,
    GeneratedResponse,
    OptimizationAction,
    ResponseStatus,
    RunPhase,
    RunResult,
    RunState,
)

EvaluatorFactory = Callable[[LLMClient], Evaluator]
OptimizerFactory = Callable[[LLMClient], Optimizer]
GeneratorFactory = Callable[[LLMClient], Generator]


class LoopOrchestrator:
    def __init__(
        self,
        config: AppConfig,
        retriever: Retriever,
        *,
        evaluator_factory: EvaluatorFactory = EvaluatorAgent,
        optimizer_factory: OptimizerFactory = OptimizerAgent,
        generator_factory: GeneratorFactory = ResponseGenerator,
        mock_scenario: str | None = None,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.evaluator_factory = evaluator_factory
        self.optimizer_factory = optimizer_factory
        self.generator_factory = generator_factory
        self.mock_scenario = mock_scenario

    async def run(self, dialogue: DialogueInput) -> RunResult:
        budget = BudgetTracker(self.config.budget)
        llm = build_llm(
            self.config.models.llm,
            usage_guard=lambda proposed: budget.check(
                llm.usage, reserved_tokens=proposed.total_tokens,
            ),
            root=self.config.project_root,
        )
        planner = QueryPlanner(llm, self.config.planning)
        generator = self.generator_factory(llm)
        evaluator = self.evaluator_factory(llm)
        optimizer = self.optimizer_factory(llm)
        policy = DecisionPolicy(self.config.quality)
        state = RunState(
            run_id=uuid4().hex, dialogue=dialogue,
            retrieval_options=self.config.retrieval.model_copy(deep=True),
            reranker_available=self.config.retrieval.rerank,
        )
        escalation: EscalationPackage | None = None

        def event(name: str, **details: object) -> None:
            record_event(state, name, budget.elapsed_ms, **details)

        def finish(
            reason: str, *, answer: str | None = None,
            status: ResponseStatus | None = None, fatal: bool = False,
        ) -> None:
            nonlocal escalation
            state.stop_reason = reason
            if status is None:
                status = (
                    ResponseStatus.ESCALATED if self.config.escalation.enabled
                    else ResponseStatus.INSUFFICIENT_EVIDENCE
                )
            if status == ResponseStatus.ESCALATED:
                escalation = package_escalation(state, reason)
                answer = (
                    "This request needs a human agent. A handoff summary has been prepared; "
                    "it has not been delivered to a live agent."
                )
            if status != ResponseStatus.ANSWERED:
                state.response = GeneratedResponse(
                    answer=answer or "The available evidence is insufficient for a reliable answer.",
                    status=status, retrieval_round=state.retrieval_round,
                )
            transition(
                state, RunPhase.FAILED if fatal else RunPhase.COMPLETED,
                budget.elapsed_ms, stop_reason=reason, status=state.response.status.value,
            )

        async def steps() -> None:
            budget.check(llm.usage)
            transition(
                state, RunPhase.PLANNING, budget.elapsed_ms,
                backend=self.config.models.llm.backend,
            )
            state.query = await planner.plan(dialogue)
            event("planning_completed", query=state.query.model_dump(mode="json"))
            next_step = "retrieve"
            while True:
                budget.check(llm.usage)
                if next_step == "retrieve":
                    state.retrieval_round += 1
                    transition(
                        state, RunPhase.RETRIEVING, budget.elapsed_ms,
                        options=state.retrieval_options.model_dump(mode="json"),
                    )
                    result = await self.retriever.retrieve(state.query, state.retrieval_options)
                    state.retrieval = result
                    state.retrieval_history.append(result.model_copy(deep=True))
                    evidence_ids = {passage.citation_id for passage in result.passages}
                    new_ids = evidence_ids - state.seen_evidence_ids
                    state.seen_evidence_ids.update(evidence_ids)
                    event(
                        "retrieval_completed", evidence_ids=sorted(evidence_ids),
                        new_evidence_ids=sorted(new_ids),
                        result=result.model_dump(mode="json"),
                    )
                    budget.check(llm.usage)
                    if (
                        state.retrieval_round > 1
                        and self.config.optimization.stop_if_no_new_evidence
                        and not new_ids
                    ):
                        finish("no_new_evidence")
                        return
                transition(
                    state, RunPhase.GENERATING, budget.elapsed_ms,
                    implementation=type(generator).__name__,
                )
                budget.check(llm.usage)
                candidate = await generator.generate(state.query, state.retrieval.passages)
                candidate = GeneratedResponse.model_validate(candidate.model_dump(mode="json"))
                validate_citations(candidate, state.retrieval.passages)
                candidate.retrieval_round = state.retrieval_round
                state.response = candidate
                state.candidate_history.append(candidate.model_copy(deep=True))
                event("generation_completed", response=candidate.model_dump(mode="json"))
                budget.check(llm.usage)
                if not self.config.quality.enabled:
                    state.stop_reason = (
                        "single_pass_mock_completed" if llm.is_mock else "single_pass_completed"
                    )
                    transition(
                        state, RunPhase.COMPLETED, budget.elapsed_ms,
                        stop_reason=state.stop_reason, status=candidate.status.value,
                        evaluated=False,
                    )
                    return
                state.iteration += 1
                transition(
                    state, RunPhase.EVALUATING, budget.elapsed_ms,
                    implementation=type(evaluator).__name__,
                )
                raw = await evaluator.evaluate(state.query, state.retrieval.passages, candidate)
                raw = EvaluationFeedback.model_validate(raw.model_dump(mode="json"))
                feedback = policy.apply(raw, candidate)
                state.feedback.append(feedback.model_copy(deep=True))
                event(
                    "evaluation_completed", raw_feedback=raw.model_dump(mode="json"),
                    feedback=feedback.model_dump(mode="json"),
                )
                budget.check(llm.usage)
                if feedback.decision == Decision.PASS:
                    finish("evaluation_passed", status=ResponseStatus.ANSWERED)
                    return
                if feedback.decision == Decision.CLARIFY:
                    finish(
                        "clarification_required", status=ResponseStatus.CLARIFICATION_NEEDED,
                        answer=feedback.clarification_question
                        or "Which service or program are you asking about?",
                    )
                    return
                if feedback.decision == Decision.ESCALATE:
                    finish("evaluator_escalation")
                    return
                if not self.config.optimization.enabled:
                    finish("optimization_disabled")
                    return
                if state.retry_count >= self.config.optimization.max_retries:
                    finish("retry_limit")
                    return
                transition(
                    state, RunPhase.OPTIMIZING, budget.elapsed_ms,
                    implementation=type(optimizer).__name__,
                )
                state.usage = list(llm.usage)
                action = await optimizer.optimize(state, feedback)
                action = OptimizationAction.model_validate(action.model_dump(mode="json"))
                budget.check(llm.usage)
                event("optimization_proposed", action=action.model_dump(mode="json"))
                next_step = apply_action(state, action)
                if next_step == "unchanged":
                    finish("no_effective_action")
                    return
                state.actions.append(action.model_copy(deep=True))
                if next_step == "clarify":
                    finish(
                        "optimizer_clarification", status=ResponseStatus.CLARIFICATION_NEEDED,
                        answer=action.clarification_question,
                    )
                    return
                if next_step == "escalate":
                    finish("optimizer_escalation")
                    return
                state.retry_count += 1
                event(
                    "optimization_applied", action=action.model_dump(mode="json"),
                    next_step=next_step, retry_count=state.retry_count,
                    query=state.query.model_dump(mode="json"),
                    options=state.retrieval_options.model_dump(mode="json"),
                )

        try:
            await asyncio.wait_for(steps(), timeout=self.config.budget.max_latency_seconds)
        except (asyncio.TimeoutError, BudgetExceeded) as exc:
            reason = str(exc) or "latency_budget"
            event("budget_stopped", reason=reason)
            finish(reason)
        except Exception as exc:
            event("component_error", error_type=type(exc).__name__, message=str(exc))
            finish(
                f"component_error:{type(exc).__name__}", fatal=True,
                status=ResponseStatus.ERROR,
                answer="The offline request could not be completed because a component failed.",
            )
        if state.response is None:
            raise RuntimeError("Orchestrator reached a terminal state without a response")
        return RunResult(
            run_id=state.run_id, conversation_id=dialogue.conversation_id,
            response=state.response, query=state.query, retrieval=state.retrieval,
            feedback=state.feedback, actions=state.actions,
            retrieval_options=state.retrieval_options,
            retrieval_history=state.retrieval_history, candidate_history=state.candidate_history,
            trace=state.trace, usage=llm.usage, elapsed_ms=budget.elapsed_ms,
            iteration=state.iteration, retrieval_round=state.retrieval_round,
            retry_count=state.retry_count, stop_reason=state.stop_reason,
            is_mock=(
                llm.is_mock or self.config.models.embedding.backend == "hash"
                or self.mock_scenario is not None
            ),
            model_backend=self.config.models.llm.backend, model_calls=llm.calls,
            escalation=escalation, mock_scenario=self.mock_scenario,
        )
