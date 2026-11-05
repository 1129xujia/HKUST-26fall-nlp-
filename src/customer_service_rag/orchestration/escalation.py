from __future__ import annotations

from customer_service_rag.schemas import EscalationPackage, RunState


def package_escalation(state: RunState, reason: str) -> EscalationPackage:
    evidence = {
        passage.citation_id: passage
        for result in state.retrieval_history for passage in result.passages
    }
    return EscalationPackage(
        reason=reason,
        conversation_id=state.dialogue.conversation_id,
        current_message=state.dialogue.current_message,
        dialogue_history=state.dialogue.dialogue_history,
        evidence_ids=sorted(state.seen_evidence_ids),
        evidence=[evidence[key] for key in sorted(evidence)],
        query=state.query,
        attempted_actions=[action.action for action in state.actions],
        missing_information=list(dict.fromkeys(
            feedback.missing_information for feedback in state.feedback
            if feedback.missing_information
        )),
        evaluation_feedback=state.feedback,
        delivered_to_human=False,
    )
