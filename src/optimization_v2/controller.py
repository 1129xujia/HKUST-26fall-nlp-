"""Execute one branch per round and re-evaluate before publishing any answer."""

from __future__ import annotations

from customer_service_rag.schemas import GeneratedResponse, QueryPlan, ResponseStatus

from .budget import BudgetExceeded, BudgetLedger
from .interfaces import QualityEvaluator, QueryRewriter, RetrievalExecutor, RevisionGenerator
from .retrieval_adapter import context_signature, fingerprint
from .router import OptimizationRouter, apply_policy
from .schemas import (
    EvaluationRequest, Feedback, GenerationRequest, OptimizationInput,
    OptimizationResult, QueryUpdate, RewriteRequest, TraceEntry,
)


def valid_citations(response, evidence):
    available = {e.passage_id for e in evidence}
    return (set(response.citations) <= available and (
        response.status != ResponseStatus.ANSWERED or bool(response.citations)
    ))


class OptimizationController:
    def __init__(self, *, retrieval: RetrievalExecutor, generator: RevisionGenerator,
                 evaluator: QualityEvaluator, rewriter: QueryRewriter | None = None,
                 router: OptimizationRouter | None = None):
        self.retrieval = retrieval
        self.generator = generator
        self.evaluator = evaluator
        self.rewriter = rewriter
        self.router = router or OptimizationRouter()

    async def run(self, request: OptimizationInput) -> OptimizationResult:
        # Revalidate public inputs and isolate every request, including cached data.
        state = OptimizationInput.model_validate(request.model_dump()).model_copy(deep=True)
        budget = BudgetLedger(state.limits, state.budget)
        if hasattr(self.retrieval, "capabilities"):
            actual = self.retrieval.capabilities
            state.capabilities.reranking &= actual.reranking
            state.capabilities.hybrid &= actual.hybrid
            state.capabilities.retrievers = [
                name for name in state.capabilities.retrievers if name in actual.retrievers
            ]
        state.capabilities.rewriting &= (
            self.rewriter is not None or bool(state.feedback.suggested_queries)
        )
        candidates = [state.response.model_copy(deep=True)]
        feedbacks = [state.feedback.model_copy(deep=True)]
        actions = [a.model_copy(deep=True) for a in state.action_history]
        trace = []
        retrieval_rounds = 0
        evaluations = 0

        def event(name, **details):
            trace.append(TraceEntry(event=name, details={
                "retry_count": state.retry_count,
                "used_tokens": budget.used_tokens,
                "elapsed_seconds": budget.elapsed_seconds,
                **details,
            }))

        def finish(reason, *, passed=False, question=None, error=False):
            escalation = None
            if passed:
                response = state.response.model_copy(deep=True)
            elif question is not None:
                response = GeneratedResponse(
                    answer=question, status=ResponseStatus.CLARIFICATION_NEEDED,
                )
            elif error:
                response = GeneratedResponse(
                    answer="A component failed while optimizing this answer.",
                    status=ResponseStatus.ERROR,
                )
            else:
                if state.limits.escalation_enabled:
                    escalation = {
                        "reason": reason, "delivered_to_human": False,
                        "dialogue": state.query.model_dump(mode="json"),
                        "evidence": [e.to_dict() for e in state.evidence],
                        "feedback": [f.model_dump(mode="json") for f in feedbacks],
                        "attempted_actions": [a.model_dump(mode="json") for a in actions],
                    }
                response = GeneratedResponse(
                    answer=("A human handoff summary has been prepared; delivery is pending."
                            if escalation else "Available evidence cannot support a reliable answer."),
                    status=(ResponseStatus.ESCALATED if escalation
                            else ResponseStatus.INSUFFICIENT_EVIDENCE),
                )
            event("completed", reason=reason, status=response.status.value)
            return OptimizationResult(
                response=response, query=state.query, evidence=state.evidence,
                options=state.options, feedback_history=feedbacks,
                candidate_history=candidates, actions=actions, trace=trace,
                retry_count=state.retry_count, retrieval_rounds=retrieval_rounds,
                evaluations=evaluations, used_tokens=budget.used_tokens,
                elapsed_seconds=budget.elapsed_seconds, stop_reason=reason,
                snapshot=state.snapshot, escalation=escalation,
            )

        event("input_received", raw_feedback=state.feedback.model_dump(mode="json"),
              query=state.query.model_dump(mode="json"),
              evidence_ids=[e.passage_id for e in state.evidence])
        try:
            while True:
                budget.check()
                state.feedback = apply_policy(state)
                feedbacks[-1] = state.feedback.model_copy(deep=True)
                action = self.router.route(state)
                event("routed", action=action.model_dump(mode="json"),
                      feedback=state.feedback.model_dump(mode="json"),
                      capabilities=state.capabilities.model_dump(),
                      options_before=state.options.model_dump(),
                      query_before=state.query.model_dump(mode="json"))
                if action.branch == "pass":
                    return finish("evaluation_passed", passed=True)
                if action.branch == "clarify":
                    return finish("clarification_required", question=action.clarification_question)
                if action.branch in {"escalate", "stop"}:
                    return finish(action.reason if action.branch == "stop" else "escalation_required")

                old_response = state.response.model_dump(mode="json")
                old_context = context_signature(state.evidence)
                old_feedback = fingerprint(state.feedback.model_dump(mode="json"))
                old_query = QueryUpdate(
                    rewritten_query=state.query.rewritten_query,
                    sub_queries=state.query.sub_queries,
                )
                revision = action.revision
                if action.branch == "rewrite_query":
                    rewrite_request = RewriteRequest(
                        query=state.query, evidence=state.evidence,
                        previous_response=state.response, feedback=state.feedback,
                        options=state.options, action_history=actions,
                    )
                    if self.rewriter is not None:
                        update = await budget.run(lambda: self.rewriter.rewrite(rewrite_request, budget))
                    else:
                        suggested = state.feedback.suggested_queries
                        update = QueryUpdate(rewritten_query=suggested[0], sub_queries=suggested[1:])
                    update = QueryUpdate.model_validate(update.model_dump())
                    before = QueryUpdate(
                        rewritten_query=state.query.rewritten_query,
                        sub_queries=state.query.sub_queries,
                    )
                    if update == before:
                        return finish("no_effective_action")
                    # Only retrieval query fields can be updated. The user's
                    # question, history, intent and trusted domain remain intact.
                    state.query = QueryPlan.model_validate({
                        **state.query.model_dump(mode="json"), **update.model_dump(),
                    })
                elif action.options is not None:
                    if action.options == state.options:
                        return finish("no_effective_action")
                    state.options = action.options.model_copy(deep=True)
                elif not state.evidence:
                    return finish("no_evidence_for_regeneration")

                state.retry_count += 1
                actions.append(action.model_copy(deep=True))
                event("action_applied", action=action.model_dump(mode="json"),
                      query=state.query.model_dump(mode="json"),
                      options=state.options.model_dump())
                if action.branch != "regenerate":
                    snapshot = await budget.run(lambda: self.retrieval.retrieve(
                        state.query, state.options, state.snapshot, state.trace_id,
                    ))
                    retrieval_rounds += int(not snapshot.reused_recall)
                    state.snapshot = snapshot.model_copy(deep=True)
                    state.evidence = list(state.snapshot.evidence)
                    event("retrieval_completed", reused_recall=snapshot.reused_recall,
                          branch_counts=snapshot.branch_counts,
                          evidence_ids=[e.passage_id for e in state.evidence],
                          context_signature=context_signature(state.evidence),
                          notes=snapshot.notes)
                    if not state.evidence:
                        # Re-evaluate the empty-evidence situation through routing
                        # rather than inventing a grounded answer.
                        state.feedback = Feedback(
                            **{**state.feedback.model_dump(mode="json"),
                               "decision": "retry", "sufficiency": 0,
                               "failure_types": ["retrieval_coverage"],
                               "evidence_sufficient": False,
                               "reason": "No evidence was retrieved"},
                        )
                        feedbacks.append(state.feedback.model_copy(deep=True))
                        continue
                    if (context_signature(state.evidence) == old_context
                            and QueryUpdate(rewritten_query=state.query.rewritten_query,
                                            sub_queries=state.query.sub_queries) == old_query):
                        return finish("no_context_change")

                generation = GenerationRequest(
                    query=state.query, evidence=state.evidence,
                    previous_response=state.response, feedback=state.feedback,
                    revision=revision,
                )
                response = await budget.run(lambda: self.generator.generate(generation, budget))
                response = GeneratedResponse.model_validate(response.model_dump())
                candidates.append(response.model_copy(deep=True))
                state.response = response
                event("generated", response=response.model_dump(mode="json"),
                      revision=revision.model_dump() if revision else None)
                if not valid_citations(response, state.evidence):
                    return finish("invalid_citations", error=True)
                evaluation = EvaluationRequest(
                    query=state.query, evidence=state.evidence, response=response,
                )
                raw = await budget.run(lambda: self.evaluator.evaluate(evaluation, budget))
                evaluations += 1
                state.feedback = Feedback.model_validate(raw.model_dump(mode="json"))
                event("evaluated", raw_feedback=state.feedback.model_dump(mode="json"))
                state.feedback = apply_policy(state)
                feedbacks.append(state.feedback.model_copy(deep=True))
                if (state.feedback.needs_optimization
                        and response.model_dump(mode="json") == old_response
                        and context_signature(state.evidence) == old_context
                        and fingerprint(state.feedback.model_dump(mode="json")) == old_feedback):
                    return finish("no_improvement")
        except BudgetExceeded as exc:
            return finish(str(exc))
        except Exception as exc:
            event("component_error", error_type=type(exc).__name__, message=str(exc))
            return finish(f"component_error:{type(exc).__name__}", error=True)
