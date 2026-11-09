"""Deterministic, inspectable routing; scores alone do not diagnose ranking."""

from customer_service_rag.schemas import Decision, ResponseStatus

from .schemas import Feedback, OptimizationAction, OptimizationInput, Revision


def apply_policy(state: OptimizationInput) -> Feedback:
    feedback = state.feedback.model_copy(deep=True)
    if feedback.decision in {Decision.CLARIFY, Decision.ESCALATE}:
        return feedback
    failures = [name for name in (
        "relevance", "sufficiency", "groundedness", "dialogue_consistency",
    ) if getattr(feedback, name) < getattr(state.limits, f"{name}_threshold")]
    valid = {item.passage_id for item in state.evidence}
    if (state.response.status != ResponseStatus.ANSWERED
            or not state.response.citations
            or not set(state.response.citations) <= valid):
        failures.append("supported_answer")
    if feedback.evidence_sufficient is False:
        failures.append("evidence_insufficient")
    if feedback.failure_types:
        failures.extend(feedback.failure_types)
    if failures:
        feedback.decision = Decision.RETRY
        feedback.reason = (
            feedback.reason.split(" Below policy:", 1)[0]
            + " Below policy: " + ", ".join(failures)
        )
    return feedback


class OptimizationRouter:
    def route(self, state: OptimizationInput) -> OptimizationAction:
        f = apply_policy(state)
        reason = f.reason.strip() or f.missing_information.strip() or "Improve the candidate"
        if f.decision == Decision.PASS:
            return OptimizationAction(branch="pass", reason=reason)
        if f.decision == Decision.ESCALATE:
            return OptimizationAction(branch="escalate", reason=reason)
        if f.decision == Decision.CLARIFY or "missing_user_information" in f.failure_types:
            return OptimizationAction(
                branch="clarify", reason=reason,
                clarification_question=f.clarification_question
                or "Which service, application type, or user situation do you mean?",
            )
        if state.retry_count >= state.limits.max_retries:
            return OptimizationAction(branch="stop", reason="retry_limit")

        # Explicit structured diagnosis > suggested action > conservative fallback.
        if f.failure_types:
            target = next((branch for failure, branch in (
                ("query_misunderstanding", "rewrite_query"),
                ("retrieval_coverage", "adjust_hybrid"),
                ("retrieval_balance", "adjust_hybrid"),
                ("evidence_ranking", "rerank"),
                ("answer_defect", "regenerate"),
            ) if failure in f.failure_types), "regenerate")
        else:
            target = {
                "rewrite_query": "rewrite_query", "add_sub_queries": "rewrite_query",
                "adjust_retrieval": "adjust_hybrid", "adjust_hybrid": "adjust_hybrid",
                "rerank": "rerank", "regenerate": "regenerate",
            }.get(f.suggested_action)
            if target is None:
                target = "adjust_hybrid" if (
                    not state.evidence or (f.evidence_sufficient is not True
                    and f.sufficiency < state.limits.sufficiency_threshold)
                ) else "regenerate"
        if target == "regenerate" and (
            not state.evidence or f.evidence_sufficient is False
        ):
            target = "adjust_hybrid"

        caps = state.capabilities
        allowed = {
            "regenerate": caps.generation,
            "rewrite_query": caps.rewriting and caps.generation,
            "adjust_hybrid": caps.hybrid and caps.generation,
            "rerank": caps.reranking and caps.generation,
        }
        if not allowed.get(target, False):
            return OptimizationAction(branch="stop", reason=f"unsupported_branch:{target}")
        if target == "regenerate":
            return OptimizationAction(branch=target, reason=reason, revision=Revision(
                instruction=("Revise the previous answer using only the supplied evidence. "
                             "Address the evaluation diagnosis and missing requirements; "
                             "remove unsupported claims and repair citations."),
                missing_information=f.missing_information, evaluation_reason=reason,
            ))
        if target == "rewrite_query":
            return OptimizationAction(branch=target, reason=reason)

        values = state.options.model_dump()
        if target == "adjust_hybrid":
            for name in ("dense", "bm25"):
                quota = f"{name}_top_k"
                if name in caps.retrievers:
                    values[quota] = min(500, max(20, values[quota] * 2))
                    values[f"{name}_weight"] = max(1.0, values[f"{name}_weight"])
                else:
                    values[quota] = 0
                    values[f"{name}_weight"] = 0
            if f.preferred_retriever in caps.retrievers:
                other = "bm25" if f.preferred_retriever == "dense" else "dense"
                values[f"{f.preferred_retriever}_weight"] = 2.0
                values[f"{other}_weight"] = 1.0 if other in caps.retrievers else 0.0
            values["final_top_k"] = min(20, values["final_top_k"] + 3)
            values["candidate_top_k"] = min(500, max(
                values["candidate_top_k"] * 2, values["final_top_k"],
            ))
        else:
            values["rerank"] = True
            if state.options.rerank:
                # A bounded strategy change, not model-generated executable code.
                values["reranker_use_sub_queries"] = not values["reranker_use_sub_queries"]
            values["candidate_top_k"] = min(500, values["candidate_top_k"] * 2)
        return OptimizationAction(branch=target, reason=reason, options=values)
