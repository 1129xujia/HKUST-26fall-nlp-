"""Validate and execute state changes proposed by an Optimizer."""

from __future__ import annotations

from typing import Literal

from customer_service_rag.schemas import OptimizationAction, QueryPlan, RunState


def apply_action(
    state: RunState, action: OptimizationAction,
) -> Literal["retrieve", "generate", "clarify", "escalate", "unchanged"]:
    if state.query is None or state.retrieval_options is None:
        raise ValueError("An optimization action requires a query and retrieval options")
    # Revalidate at the execution boundary; do not trust an injected implementation.
    action = OptimizationAction.model_validate(action.model_dump(mode="json"))
    if action.action in {"clarify", "escalate"}:
        return action.action
    if action.action == "regenerate":
        if state.retrieval is None or not state.retrieval.passages:
            raise ValueError("Cannot regenerate without evidence; retrieve again instead")
        return "generate"
    if action.action == "adjust_retrieval":
        options = action.retrieval_options
        if options.rerank and not state.reranker_available:
            raise ValueError("An optimizer cannot enable an uninitialized reranker")
        if options == state.retrieval_options:
            return "unchanged"
        state.retrieval_options = options.model_copy(deep=True)
        return "retrieve"
    values = state.query.model_dump(mode="json")
    if action.action == "rewrite_query":
        if " ".join(action.rewritten_query.casefold().split()) == " ".join(
            state.query.rewritten_query.casefold().split()
        ):
            return "unchanged"
        values["rewritten_query"] = action.rewritten_query
    elif action.action == "add_sub_queries":
        existing = {text.casefold().strip() for _, text in state.query.retrieval_queries()}
        additions = []
        for text in action.sub_queries:
            key = text.casefold().strip()
            if key not in existing:
                existing.add(key)
                additions.append(text)
        if not additions:
            return "unchanged"
        values["sub_queries"] = [*state.query.sub_queries, *additions]
    state.query = QueryPlan.model_validate(values)
    return "retrieve"
