"""Conservative lexical development rules, not a semantic evaluator or real LLM."""

from __future__ import annotations

import re
from typing import Any

from customer_service_rag.models.embeddings import tokenize

STOPWORDS = set(
    "a an the i me my you your we our they their it that this these those "
    "is are was were be been do does did to for from with of on in at as "
    "and or but if can could would should will how what which where when "
    "please need want about again have has had am may".split()
)


def content_words(text: str) -> set[str]:
    return set(tokenize(text)) - STOPWORDS


def infer_service(query: dict[str, Any]) -> tuple[str | None, str]:
    cues = [
        ("studentaid", "student aid", ("student aid", "financial aid", "international student")),
        ("dmv", "driving license", ("license", "dl-r1", "dl-l2", "dmv")),
        ("va", "veteran support", ("veteran", "va-a1")),
        ("ssa", "benefit application", ("benefit", "bank", "payment")),
    ]
    texts = [query["current_message"]] + [
        message["text"] for message in reversed(query["dialogue_history"])
    ]
    for text in texts:
        for domain, label, terms in cues:
            if any(term in text.casefold() for term in terms):
                return domain, label
    return None, ""


def evaluate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    query, response, passages = payload["query"], payload["response"], payload["passages"]
    message = query["current_message"].casefold()
    domain, service = infer_service(query)
    result: dict[str, Any] = {
        "relevance": 0.9, "sufficiency": 0.9, "groundedness": 1.0,
        "dialogue_consistency": 0.9, "decision": "pass",
        "reason": "Mock lexical checks found no blocking issue; not a semantic quality guarantee.",
    }
    if any(term in message for term in ("human", "live agent", "representative")):
        return {
            **result, "decision": "escalate",
            "reason": "The user explicitly requested a human agent.",
            "suggested_action": "escalate",
        }
    if re.search(r"\btransfer\b.*\b(money|funds|payment|benefit)\b", message):
        return {
            **result, "decision": "escalate",
            "reason": "The requested account transaction cannot be executed by this knowledge assistant.",
            "suggested_action": "escalate",
        }
    if domain is None and not query["dialogue_history"]:
        return {
            **result, "relevance": 0.2, "sufficiency": 0.0, "decision": "clarify",
            "missing_information": "The service or program the user means is not specified.",
            "clarification_question": "Which service or program are you asking about?",
            "suggested_action": "clarify", "reason": "Insufficient user context.",
        }
    if not passages:
        return {
            **result, "relevance": 0.0, "sufficiency": 0.0, "groundedness": 0.0,
            "decision": "retry", "missing_information": "No supporting evidence was retrieved.",
            "suggested_action": "adjust_retrieval", "reason": "Empty evidence set.",
        }
    by_id = {f'{p["doc_id"]}:{p["passage_id"]}': p for p in passages}
    cited = [by_id[key] for key in response["citations"] if key in by_id]
    if response["status"] != "answered" or not cited:
        return {
            **result, "relevance": 0.4, "groundedness": 0.0, "decision": "retry",
            "missing_information": "A supported, cited answer has not been generated.",
            "suggested_action": "regenerate", "reason": "Incomplete candidate response.",
        }
    if domain and any(p["domain"] != domain for p in cited):
        return {
            **result, "relevance": 0.1, "dialogue_consistency": 0.2, "decision": "retry",
            "missing_information": f"Evidence about {service} is required by the visible dialogue.",
            "suggested_action": "add_sub_queries",
            "suggested_queries": [f"{service}: {query['current_message']}"],
            "reason": "The cited evidence belongs to a different service.",
        }
    answer_words = content_words(response["answer"])
    cited_words = content_words(" ".join(p["text"] for p in cited))
    result["groundedness"] = len(answer_words & cited_words) / max(1, len(answer_words))
    visible_text = query["current_message"] + " " + " ".join(
        item["text"] for item in query["dialogue_history"]
    )
    needed_words = content_words(visible_text)
    overlap = len(needed_words & answer_words) / max(1, len(needed_words))
    result["relevance"] = min(0.95, 0.55 + overlap)
    requirements = []
    if "deadline" in message or "when" in message:
        requirements.append({"deadline", "within", "days"})
    if "documents" in message or "bring" in message:
        requirements.append({"documents", "confirmation", "identification"})
    evidence_words = content_words(" ".join(p["text"] for p in passages))
    if any(not (terms & evidence_words) for terms in requirements):
        result.update(
            sufficiency=0.4, decision="retry",
            missing_information="A requested evidence topic is missing.",
            suggested_action="adjust_retrieval", reason="Mock evidence coverage check failed.",
        )
    elif any(not (terms & answer_words) for terms in requirements):
        result.update(
            relevance=0.4, decision="retry",
            missing_information="The answer omits a requested topic available in the evidence.",
            suggested_action="regenerate", reason="Mock answer coverage check failed.",
        )
    elif result["groundedness"] < 0.8:
        result.update(
            decision="retry", suggested_action="regenerate",
            missing_information="Some answer terms are absent from the cited evidence.",
            reason="Mock lexical support check failed.",
        )
    return result


def optimize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    feedback = payload["feedback"]
    reason = feedback.get("reason") or feedback.get("missing_information") or "Improve the candidate."
    if feedback["decision"] == "clarify":
        return {
            "action": "clarify", "reason": reason,
            "clarification_question": feedback.get("clarification_question")
            or "Which service or program are you asking about?",
        }
    if feedback["decision"] == "escalate":
        return {"action": "escalate", "reason": reason}
    if feedback.get("suggested_action") == "regenerate":
        return {"action": "regenerate", "reason": reason}
    queries = feedback.get("suggested_queries", [])
    if queries:
        if feedback.get("suggested_action") == "rewrite_query":
            return {"action": "rewrite_query", "rewritten_query": queries[0], "reason": reason}
        return {"action": "add_sub_queries", "sub_queries": queries, "reason": reason}
    options = dict(payload["retrieval_options"])
    options["branches"] = list(dict.fromkeys([*options["branches"], "dense", "bm25"]))
    options["per_branch_top_k"] = min(100, options["per_branch_top_k"] * 2)
    options["final_top_k"] = min(20, options["final_top_k"] + 3)
    options["candidate_top_k"] = max(options["candidate_top_k"], options["final_top_k"])
    return {"action": "adjust_retrieval", "reason": reason, "retrieval_options": options}
