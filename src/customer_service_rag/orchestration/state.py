from __future__ import annotations

from customer_service_rag.schemas import RunPhase, RunState, TraceEvent

TRANSITIONS = {
    RunPhase.CREATED: {RunPhase.PLANNING, RunPhase.COMPLETED, RunPhase.FAILED},
    RunPhase.PLANNING: {RunPhase.RETRIEVING, RunPhase.COMPLETED, RunPhase.FAILED},
    RunPhase.RETRIEVING: {RunPhase.GENERATING, RunPhase.COMPLETED, RunPhase.FAILED},
    RunPhase.GENERATING: {RunPhase.EVALUATING, RunPhase.COMPLETED, RunPhase.FAILED},
    RunPhase.EVALUATING: {RunPhase.OPTIMIZING, RunPhase.COMPLETED, RunPhase.FAILED},
    RunPhase.OPTIMIZING: {
        RunPhase.RETRIEVING, RunPhase.GENERATING, RunPhase.COMPLETED, RunPhase.FAILED,
    },
    RunPhase.COMPLETED: set(),
    RunPhase.FAILED: set(),
}


def transition(
    state: RunState, target: RunPhase, elapsed_ms: float, **details: object,
) -> None:
    if target not in TRANSITIONS[state.phase]:
        raise ValueError(f"Invalid state transition: {state.phase.value} -> {target.value}")
    state.phase = target
    state.trace.append(TraceEvent(
        phase=target, iteration=state.iteration, retrieval_round=state.retrieval_round,
        retry_count=state.retry_count, elapsed_ms=elapsed_ms, details=details,
    ))


def record_event(
    state: RunState, event: str, elapsed_ms: float, **details: object,
) -> None:
    """Record a stage result without introducing a state transition."""
    state.trace.append(TraceEvent(
        event=event, phase=state.phase, iteration=state.iteration,
        retrieval_round=state.retrieval_round, retry_count=state.retry_count,
        elapsed_ms=elapsed_ms, details=details,
    ))
