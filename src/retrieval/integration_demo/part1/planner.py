"""A small, dependency-free Part 1 query planner for integration tests."""

from dataclasses import asdict, dataclass, field
import re


DOMAIN_TERMS = {
    "dmv": ("driver", "license", "vehicle", "dmv", "registration", "permit"),
    "ssa": ("social security", "retirement", "survivor", "disability", "medicare"),
    "va": ("veteran", "va ", "military", "service member"),
    "studentaid": ("student aid", "fafsa", "loan", "grant", "college", "school"),
}


@dataclass(frozen=True)
class QueryPlan:
    query: str
    sub_queries: list[str] = field(default_factory=list)
    domain: str | None = None
    context_query: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class RuleBasedPlanner:
    """Resolve light conversational context and propose search variants."""

    def plan(self, question: str, history: list[dict[str, str]] | None = None) -> QueryPlan:
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")
        history = history or []
        previous_user = next(
            (item.get("content", "").strip() for item in reversed(history) if item.get("role") == "user"),
            "",
        )
        lowered = question.lower()
        needs_context = bool(re.search(r"\b(it|that|this|they|those|them)\b", lowered))
        context_query = f"{previous_user} {question}".strip() if needs_context and previous_user else question
        domain = self._detect_domain(context_query)
        sub_queries = self._expand(question, context_query)
        return QueryPlan(
            query=context_query,
            sub_queries=sub_queries,
            domain=domain,
            context_query=context_query,
        )

    @staticmethod
    def _detect_domain(text: str) -> str | None:
        lowered = text.lower()
        scored = {
            domain: sum(term in lowered for term in terms)
            for domain, terms in DOMAIN_TERMS.items()
        }
        domain, score = max(scored.items(), key=lambda item: item[1])
        return domain if score else None

    @staticmethod
    def _expand(question: str, context_query: str) -> list[str]:
        variants: list[str] = []
        if context_query != question:
            variants.append(question)
        lowered = question.lower()
        if any(word in lowered for word in ("how", "apply", "get", "renew")):
            variants.append(f"requirements and steps for {question}")
        if any(word in lowered for word in ("cost", "fee", "pay")):
            variants.append(f"fees and payment for {question}")
        if any(word in lowered for word in ("eligible", "qualify", "eligibility")):
            variants.append(f"eligibility requirements for {question}")
        seen: set[str] = {context_query}
        result = []
        for item in variants:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result[:3]
