from retrieval.integration_demo.orchestrator import EndToEndDemo
from retrieval.integration_demo.part1 import RuleBasedPlanner
from retrieval.integration_demo.part3 import AnswerEvaluator, ExtractiveCitedGenerator
from retrieval.schemas import Evidence


def evidence(identifier="p"):
    return Evidence(identifier, "d", "dmv", "License", "Renewal", "Bring proof of identity. Then pay the fee.", ["1"], ["bm25"], "q")


class Service:
    def __init__(self, empty_first=False):
        self.calls = []
        self.empty_first = empty_first

    def retrieve(self, request):
        self.calls.append(request)
        if self.empty_first and len(self.calls) == 1:
            return []
        return [evidence()]


def test_planner_detects_domain_and_expands() -> None:
    plan = RuleBasedPlanner().plan("How do I renew my driver license?")
    assert plan.domain == "dmv"
    assert plan.sub_queries


def test_planner_resolves_pronoun_from_history() -> None:
    plan = RuleBasedPlanner().plan("How much does it cost?", [{"role": "user", "content": "renew driver license"}])
    assert "driver license" in plan.query


def test_generator_and_evaluator_keep_valid_citations() -> None:
    items = [evidence()]
    answer = ExtractiveCitedGenerator().generate("question", items)
    assert AnswerEvaluator().evaluate(answer, items)["grounded"] is True


def test_end_to_end_result() -> None:
    result = EndToEndDemo(Service()).run("How do I renew my driver license?")
    assert result.quality["grounded"] is True
    assert result.evidence[0]["passage_id"] == "p"


def test_domain_retry_is_visible() -> None:
    service = Service(empty_first=True)
    result = EndToEndDemo(service).run("How do I renew my driver license?")
    assert result.retried_without_domain is True
    assert service.calls[1].domain is None


def test_empty_evidence_has_safe_answer() -> None:
    class Empty:
        def retrieve(self, request): return []
    result = EndToEndDemo(Empty()).run("general question")
    assert result.quality["grounded"] is False
    assert "could not find" in result.answer.text
