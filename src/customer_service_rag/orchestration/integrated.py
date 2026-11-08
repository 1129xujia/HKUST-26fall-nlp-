"""Unified CLI runner for Part 2, Part 3 and optimization_v2."""

from __future__ import annotations

from uuid import uuid4

from customer_service_rag.models.llm import build_llm
from customer_service_rag.part3 import EvidenceResponseGenerator
from customer_service_rag.planning.planner import QueryPlanner
from customer_service_rag.schemas import (
    EscalationPackage, EvaluationFeedback, GeneratedResponse, OptimizationAction,
    ResponseStatus, RunPhase, RunResult, TraceEvent,
)
from optimization_v2.adapters import LLMQueryRewriter, LLMRevisionGenerator, MainEvaluatorAdapter
from optimization_v2.budget import BudgetExceeded, BudgetLedger, BudgetedLLM
from optimization_v2.controller import OptimizationController
from optimization_v2.schemas import (
    BudgetState, EvaluationRequest, Limits, OptimizationInput,
)

from .integrated_retrieval import IntegratedRetriever, legacy_options


class RecordingRetrieval:
    """A request-local observer; shared model/index instances stay immutable."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.capabilities = adapter.capabilities
        self.snapshots = []

    async def retrieve(self, *args, **kwargs):
        snapshot = await self.adapter.retrieve(*args, **kwargs)
        self.snapshots.append(snapshot.model_copy(deep=True))
        return snapshot


class IntegratedOrchestrator:
    def __init__(self, config):
        self.config = config
        self.retriever = IntegratedRetriever(config)

    async def run(self, dialogue) -> RunResult:
        config = self.config
        limits = Limits(
            max_retries=config.optimization.max_retries if config.optimization.enabled else 0,
            max_total_tokens=config.budget.max_total_tokens,
            max_latency_seconds=config.budget.max_latency_seconds,
            escalation_enabled=config.escalation.enabled,
            **{name: getattr(config.quality, name) for name in (
                "relevance_threshold", "sufficiency_threshold", "groundedness_threshold",
                "dialogue_consistency_threshold",
            )},
        )
        budget = BudgetLedger(limits, BudgetState())
        llm = build_llm(config.models.llm, root=config.project_root)
        maximum = config.models.llm.max_new_tokens
        bounded = BudgetedLLM(llm, budget, maximum)
        retrieval = RecordingRetrieval(self.retriever.adapter)
        options = self.retriever.options(config.retrieval)
        trace, feedbacks, candidates, actions = [], [], [], []
        query, response, optimization = None, None, None
        retry_count, evaluations = 0, 0
        stop_reason = "not_started"
        escalation = None

        def event(phase, name, *, elapsed_ms=None, **details):
            values = dict(phase=phase, iteration=evaluations,
                          retrieval_round=details.get("retrieval_round", sum(
                              not s.reused_recall for s in retrieval.snapshots)),
                          retry_count=details.get("retry_count", retry_count),
                          elapsed_ms=elapsed_ms if elapsed_ms is not None else budget.elapsed_seconds * 1000)
            if not trace or trace[-1].phase != phase:
                trace.append(TraceEvent(event="transition", **values))
            trace.append(TraceEvent(event=name, details=details, **values))

        def make_handoff(reason):
            return EscalationPackage(
                reason=reason, conversation_id=dialogue.conversation_id,
                current_message=dialogue.current_message, dialogue_history=dialogue.dialogue_history,
                query=query, evidence=list(self.retriever.result(retrieval.snapshots[-1]).passages)
                if retrieval.snapshots else [],
                evidence_ids=[e.passage_id for e in retrieval.snapshots[-1].evidence]
                if retrieval.snapshots else [],
                attempted_actions=[a.action for a in actions], evaluation_feedback=feedbacks,
                missing_information=[f.missing_information for f in feedbacks if f.missing_information],
            )

        event(RunPhase.CREATED, "input_received")
        try:
            event(RunPhase.PLANNING, "planning_started")
            query = await budget.run(lambda: QueryPlanner(bounded, config.planning).plan(dialogue))
            event(RunPhase.RETRIEVING, "retrieval_started")
            snapshot = await budget.run(lambda: retrieval.retrieve(query, options))
            event(RunPhase.GENERATING, "generation_started")
            response = await budget.run(lambda: EvidenceResponseGenerator(bounded).generate(
                query, snapshot.evidence,
            ))
            response.retrieval_round = 1
            candidates.append(response.model_copy(deep=True))
            if not config.quality.enabled:
                stop_reason = "single_pass_completed"
            else:
                event(RunPhase.EVALUATING, "evaluation_started")
                evaluator = MainEvaluatorAdapter(llm, max_output_tokens=maximum)
                feedback = await budget.run(lambda: evaluator.evaluate(
                    EvaluationRequest(query=query, evidence=snapshot.evidence, response=response), budget,
                ))
                evaluations = 1
                controller = OptimizationController(
                    retrieval=retrieval, evaluator=evaluator,
                    generator=LLMRevisionGenerator(llm, max_output_tokens=maximum),
                    rewriter=LLMQueryRewriter(llm, max_output_tokens=maximum),
                )
                optimization = await controller.run(OptimizationInput(
                    query=query, response=response, evidence=snapshot.evidence,
                    feedback=feedback, options=options, snapshot=snapshot,
                    capabilities=retrieval.capabilities, limits=limits,
                    budget=BudgetState(used_tokens=budget.used_tokens,
                                       elapsed_seconds=budget.elapsed_seconds),
                    trace_id=dialogue.conversation_id,
                ), budget=budget)
                response, query, options = optimization.response, optimization.query, optimization.options
                stop_reason, retry_count = optimization.stop_reason, optimization.retry_count
                candidates = optimization.candidate_history
                feedbacks = [EvaluationFeedback.model_validate(f.model_dump(
                    include=set(EvaluationFeedback.model_fields), mode="json",
                )) for f in optimization.feedback_history]
                applied = [e for e in optimization.trace if e.event == "action_applied"]
                for action, applied_event in zip(optimization.actions, applied):
                    if action.branch == "regenerate":
                        converted = OptimizationAction(action="regenerate", reason=action.reason)
                    elif action.branch == "rewrite_query":
                        converted = OptimizationAction(
                            action="rewrite_query", reason=action.reason,
                            rewritten_query=applied_event.details["query"]["rewritten_query"],
                        )
                    else:
                        converted = OptimizationAction(
                            action="adjust_retrieval", reason=action.reason,
                            retrieval_options=legacy_options(action.options),
                        )
                    actions.append(converted)
                # Replay native events with their original elapsed times and diagnostics.
                event_phases = {
                    "routed": RunPhase.EVALUATING, "action_applied": RunPhase.OPTIMIZING,
                    "retrieval_completed": RunPhase.RETRIEVING, "generated": RunPhase.GENERATING,
                    "evaluated": RunPhase.EVALUATING, "no_context_change": RunPhase.OPTIMIZING,
                    "component_error": RunPhase.FAILED,
                }
                evaluations = 1
                replay_round = 1
                for native in optimization.trace:
                    if native.event not in event_phases:
                        continue
                    if native.event == "evaluated":
                        evaluations += 1
                    if (native.event == "retrieval_completed"
                            and not native.details["reused_recall"]):
                        replay_round += 1
                    event(event_phases[native.event], native.event,
                          elapsed_ms=native.details["elapsed_seconds"] * 1000,
                          retrieval_round=replay_round,
                          **native.details)
                evaluations = 1 + optimization.evaluations
                if optimization.escalation is not None:
                    escalation = make_handoff(stop_reason)
        except BudgetExceeded as exc:
            stop_reason = str(exc)
            if config.escalation.enabled:
                escalation = make_handoff(stop_reason)
            response = GeneratedResponse(
                answer="The request budget was exhausted; a reliable answer could not be completed.",
                status=ResponseStatus.ESCALATED if escalation else ResponseStatus.INSUFFICIENT_EVIDENCE,
            )
        except Exception as exc:
            stop_reason = f"component_error:{type(exc).__name__}"
            event(RunPhase.FAILED, "component_error", error_type=type(exc).__name__, message=str(exc))
            response = GeneratedResponse(answer="A component failed while handling the request.",
                                         status=ResponseStatus.ERROR)
        event(RunPhase.FAILED if response.status == ResponseStatus.ERROR else RunPhase.COMPLETED,
              "completed", stop_reason=stop_reason)
        history = [self.retriever.result(s) for s in retrieval.snapshots]
        return RunResult(
            run_id=uuid4().hex, conversation_id=dialogue.conversation_id, query=query,
            response=response, retrieval=history[-1] if history else None,
            retrieval_history=history, retrieval_options=legacy_options(options),
            feedback=feedbacks, actions=actions, candidate_history=candidates, trace=trace,
            usage=list(llm.usage), elapsed_ms=budget.elapsed_seconds * 1000,
            iteration=evaluations, retrieval_round=sum(not s.reused_recall for s in retrieval.snapshots),
            retry_count=retry_count, stop_reason=stop_reason, escalation=escalation,
            is_mock=llm.is_mock or config.models.embedding.backend == "hash",
            model_backend=config.models.llm.backend, model_calls=list(getattr(llm, "calls", [])),
            pipeline_backend="integrated", integration={
                "components": ["retrieval", "customer_service_rag.part3", "optimization_v2"],
                "index_backend": "shared_numpy", "citation_scheme": "shared_corpus_citation_id",
                "used_tokens": budget.used_tokens,
                "options": options.model_dump(mode="json"),
                "feedback": [f.model_dump(mode="json") for f in optimization.feedback_history]
                if optimization else [],
                "actions": [a.model_dump(mode="json") for a in optimization.actions]
                if optimization else [],
                "quality_mode": "disabled" if not config.quality.enabled else (
                    "mock_lexical" if llm.is_mock else "llm_judge_v2"
                ),
            },
        )
