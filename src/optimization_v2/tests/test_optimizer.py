from __future__ import annotations

import asyncio
import copy
import unittest
from dataclasses import asdict

import numpy as np
from pydantic import ValidationError

from customer_service_rag.schemas import (
    GeneratedResponse, LLMResult, QueryPlan, RunState, TokenUsage,
    RetrievalOptions, RetrievalResult,
)
from optimization_v2 import (
    BudgetState, Capabilities, ExecutionOptions, Feedback, Limits,
    LLMQueryRewriter, LLMRevisionGenerator, MainEvaluatorAdapter,
    OptimizationController, OptimizationInput, OptimizationRouter,
    Part2RetrievalAdapter, from_main_state,
)
from optimization_v2.adapters import to_main_passages
from optimization_v2.budget import BudgetExceeded, BudgetLedger
from optimization_v2.retrieval_adapter import context_signature
from optimization_v2.router import apply_policy
from optimization_v2.schemas import (
    EvaluationRequest, GenerationRequest, QueryUpdate, Revision, RewriteRequest,
)
from retrieval.config import RetrievalConfig
from retrieval.pipeline import HybridRetrievalPipeline
from retrieval.reranker import CrossEncoderReranker
from retrieval.schemas import Evidence, Passage, SearchHit


def passage(name):
    return Passage(
        passage_id=name, doc_id="doc:dmv", domain="dmv", title="License",
        section_title="Replacement", parent_titles=[],
        text=f"Evidence {name}: apply with identification within 30 days.", span_ids=[name],
    )


def evidence(name="p::1"):
    p = passage(name)
    return Evidence(
        passage_id=p.passage_id, doc_id=p.doc_id, domain=p.domain, title=p.title,
        section_title=p.section_title, text=p.text, span_ids=p.span_ids,
        source_retrievers=["dense"], matched_query="replace license",
    )


def feedback(**changes):
    return Feedback(**{
        "relevance": 0.9, "sufficiency": 0.9, "groundedness": 0.9,
        "dialogue_consistency": 0.9, "decision": "retry",
        "reason": "Answer omits the deadline", "suggested_action": "regenerate",
        **changes,
    })


def make_input(**changes):
    return OptimizationInput(**{
        "query": QueryPlan(
            conversation_id="conv", current_message="What do I bring and when?",
            rewritten_query="replace license documents deadline", domain="unknown",
        ),
        "response": GeneratedResponse(answer="Bring identification", citations=["p::1"]),
        "evidence": [evidence()], "feedback": feedback(),
        "options": ExecutionOptions(dense_top_k=2, bm25_top_k=2,
                                    candidate_top_k=3, final_top_k=2),
        **changes,
    })


class Search:
    def __init__(self, name, ids):
        self.name, self.ids, self.calls = name, ids, []

    def search(self, query, *, domain=None, top_k=20):
        self.calls.append((query, domain, top_k))
        return [SearchHit(passage(pid), self.name, i, 1 / i, query)
                for i, pid in enumerate(self.ids[:top_k], 1)
                if domain is None or domain == "dmv"]


class ReverseReranker:
    def rerank(self, query, sub_queries, candidates):
        for item in candidates:
            item.rerank_score = float(ord(item.passage.passage_id[-1]))
        return sorted(candidates, key=lambda c: -c.rerank_score)


def retrieval_service(*, reverse=False):
    dense = Search("dense", ["a", "b", "c", "d"])
    bm25 = Search("bm25", ["b", "a", "d", "c"])
    pipeline = HybridRetrievalPipeline(
        dense_retriever=dense, bm25_retriever=bm25,
        config=RetrievalConfig(dense_top_k=2, bm25_top_k=2, fused_top_k=4),
    )
    adapter = Part2RetrievalAdapter(
        pipeline, corpus_version="test-v1",
        rerankers={"current_first_user_history": ReverseReranker()} if reverse else None,
    )
    return adapter, dense, bm25


class Generator:
    def __init__(self, *, invalid=False, same=False):
        self.requests = []
        self.invalid, self.same = invalid, same

    async def generate(self, request, budget):
        self.requests.append(request.model_copy(deep=True))
        if self.same:
            return request.previous_response
        return GeneratedResponse(
            answer="Bring identification and apply within 30 days.",
            citations=["invented"] if self.invalid else [request.evidence[0].passage_id],
        )


class Evaluator:
    def __init__(self, *, retry=False, delay=0):
        self.retry, self.delay, self.calls = retry, delay, 0

    async def evaluate(self, request, budget):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return feedback(decision="retry" if self.retry else "pass")


class Rewriter:
    async def rewrite(self, request, budget):
        return QueryUpdate(rewritten_query="new licensing documents and deadline")


class RecordingLLM:
    def __init__(self, value=None, tokens=7, fail=False):
        self.value, self.tokens, self.fail = value, tokens, fail
        self.calls = []

    async def complete(self, prompt, *, task, payload):
        import json

        self.calls.append((prompt, task, payload))
        if self.fail:
            raise ValueError("provider failed after starting")
        return LLMResult(
            text=json.dumps(self.value), model="offline-test", is_mock=True,
            usage=TokenUsage(input_tokens=self.tokens, output_tokens=1),
        )


class ContractAndRouterTests(unittest.TestCase):
    def test_main_feedback_is_accepted_without_new_fields(self):
        original = feedback().model_dump(exclude={
            "failure_types", "evidence_sufficient", "preferred_retriever",
        })
        self.assertTrue(Feedback.model_validate(original).needs_optimization)
        self.assertNotIn("needs_optimization", Feedback.model_validate(original).model_dump())

    def test_request_json_round_trip(self):
        state = make_input()
        result = OptimizationInput.model_validate_json(state.model_dump_json())
        self.assertEqual(result.evidence[0].passage_id, "p::1")

    def test_bounds_and_nan_are_rejected(self):
        for kwargs in ({"dense_weight": float("nan")}, {"final_top_k": 21},
                       {"dense_top_k": 0, "bm25_top_k": 0},
                       {"candidate_top_k": 1, "final_top_k": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                ExecutionOptions(**kwargs)

    def test_false_pass_is_blocked_and_policy_is_idempotent(self):
        state = make_input(feedback=feedback(decision="pass", groundedness=0.1))
        state.feedback = apply_policy(state)
        self.assertEqual(state.feedback.decision.value, "retry")
        self.assertEqual(apply_policy(state), state.feedback)

    def test_invalid_citation_cannot_pass(self):
        state = make_input(feedback=feedback(decision="pass"),
                           response=GeneratedResponse(answer="x", citations=["missing"]))
        self.assertNotEqual(OptimizationRouter().route(state).branch, "pass")

    def test_terminal_decisions_precede_retries(self):
        for decision in ("pass", "clarify", "escalate"):
            state = make_input(feedback=feedback(decision=decision), retry_count=2)
            self.assertEqual(OptimizationRouter().route(state).branch, decision)

    def test_escalate_precedes_diagnosis(self):
        state = make_input(feedback=feedback(
            decision="escalate", failure_types=["missing_user_information"],
        ))
        self.assertEqual(OptimizationRouter().route(state).branch, "escalate")

    def test_diagnosis_precedes_suggestion_and_has_stable_priority(self):
        state = make_input(feedback=feedback(
            failure_types=["answer_defect", "query_misunderstanding", "retrieval_coverage"],
        ))
        self.assertEqual(OptimizationRouter().route(state).branch, "rewrite_query")

    def test_all_four_routes(self):
        for failure, expected in (
            ("answer_defect", "regenerate"), ("query_misunderstanding", "rewrite_query"),
            ("retrieval_balance", "adjust_hybrid"), ("evidence_ranking", "rerank"),
        ):
            state = make_input(feedback=feedback(failure_types=[failure]),
                               capabilities=Capabilities(reranking=True))
            self.assertEqual(OptimizationRouter().route(state).branch, expected)

    def test_insufficient_evidence_is_not_regenerated(self):
        state = make_input(feedback=feedback(evidence_sufficient=False))
        self.assertEqual(OptimizationRouter().route(state).branch, "adjust_hybrid")

    def test_unsupported_rerank_stops_explicitly(self):
        state = make_input(feedback=feedback(failure_types=["evidence_ranking"]))
        action = OptimizationRouter().route(state)
        self.assertEqual(action.reason, "unsupported_branch:rerank")

    def test_query_normalization_and_bound(self):
        value = QueryUpdate(rewritten_query="  a   b ", sub_queries=[
            "A B", " c ", "C", "d", "e", "f", "g", " ",
        ])
        self.assertEqual(value.sub_queries, ["c", "d", "e", "f"])

    def test_main_citation_mapping_is_bijective(self):
        original = [evidence("doc::1"), evidence("doc:1")]
        passages, mapping = to_main_passages(original)
        self.assertEqual(len(mapping), 2)
        self.assertEqual([mapping[p.citation_id] for p in passages], ["doc::1", "doc:1"])

    def test_context_signature_detects_order_changes(self):
        items = [evidence("a"), evidence("b")]
        self.assertNotEqual(context_signature(items), context_signature(items[::-1]))

    def test_main_state_adapter_preserves_question_ids_and_costs(self):
        passages, _ = to_main_passages([evidence()])
        source = RunState(
            run_id="run", dialogue=make_input().query,
            query=make_input().query,
            response=GeneratedResponse(answer="old", citations=[passages[0].citation_id]),
            retrieval=RetrievalResult(passages=passages),
            retrieval_options=RetrievalOptions(),
            usage=[TokenUsage(input_tokens=10, output_tokens=3)],
        )
        result = from_main_state(source, feedback())
        self.assertEqual(result.budget.used_tokens, 13)
        self.assertEqual(result.response.citations, [result.evidence[0].passage_id])


class RetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_weight_change_reuses_raw_recall_and_changes_order(self):
        adapter, dense, bm25 = retrieval_service()
        state = make_input()
        first = await adapter.retrieve(state.query, state.options)
        changed = ExecutionOptions(**{**state.options.model_dump(), "bm25_weight": 2})
        second = await adapter.retrieve(state.query, changed, first)
        self.assertEqual([e.passage_id for e in first.evidence], ["a", "b"])
        self.assertEqual([e.passage_id for e in second.evidence], ["b", "a"])
        self.assertTrue(second.reused_recall)
        self.assertEqual((len(dense.calls), len(bm25.calls)), (1, 1))

    async def test_query_quota_domain_and_corpus_invalidate_cache(self):
        adapter, dense, _ = retrieval_service()
        state = make_input()
        first = await adapter.retrieve(state.query, state.options)
        query = state.query.model_copy(update={"rewritten_query": "new query"})
        self.assertFalse((await adapter.retrieve(query, state.options, first)).reused_recall)
        changed = ExecutionOptions(**{**state.options.model_dump(), "dense_top_k": 3})
        self.assertFalse((await adapter.retrieve(state.query, changed, first)).reused_recall)
        adapter.corpus_version = "test-v2"
        self.assertFalse((await adapter.retrieve(state.query, state.options, first)).reused_recall)
        self.assertEqual(len(dense.calls), 4)

    async def test_domain_is_hint_by_default_and_unknown_maps_to_none(self):
        adapter, dense, _ = retrieval_service()
        state = make_input()
        await adapter.retrieve(state.query, state.options)
        self.assertIsNone(dense.calls[-1][1])
        query = state.query.model_copy(update={"domain": "dmv"})
        trusted = ExecutionOptions(**{**state.options.model_dump(), "domain_filter": True})
        await adapter.retrieve(query, trusted)
        self.assertEqual(dense.calls[-1][1], "dmv")
        query.domain = "unknown"
        await adapter.retrieve(query, trusted)
        self.assertIsNone(dense.calls[-1][1])

    async def test_rerank_reuses_pool_without_mutating_original(self):
        adapter, dense, _ = retrieval_service(reverse=True)
        state = make_input()
        first = await adapter.retrieve(state.query, state.options)
        original = copy.deepcopy(asdict(first.candidates[0]))
        cfg = adapter.pipeline.config
        changed = ExecutionOptions(**{**state.options.model_dump(), "rerank": True})
        second = await adapter.retrieve(state.query, changed, first)
        self.assertEqual([e.passage_id for e in second.evidence], ["b", "a"])
        self.assertEqual(len(dense.calls), 1)
        self.assertEqual(asdict(first.candidates[0]), original)
        self.assertIs(adapter.pipeline.config, cfg)
        self.assertFalse(adapter.pipeline.config.use_reranker)

    async def test_cross_encoder_wrapper_shares_model_not_settings(self):
        class Model:
            def predict(self, pairs, **kwargs):
                return np.arange(len(pairs), dtype=float)

        adapter, _, _ = retrieval_service()
        original = CrossEncoderReranker("fake", model=Model(), query_strategy="identity")
        adapter.pipeline.reranker = original
        state = make_input()
        options = ExecutionOptions(**{**state.options.model_dump(), "rerank": True})
        result = await adapter.retrieve(state.query, options)
        self.assertTrue(result.evidence)
        self.assertEqual(original.query_strategy, "identity")
        self.assertTrue(original.use_sub_queries)

    async def test_concurrent_requests_do_not_share_options(self):
        adapter, _, _ = retrieval_service()
        state = make_input()
        a = state.options
        b = ExecutionOptions(**{**a.model_dump(), "bm25_weight": 3})
        first, second = await asyncio.gather(
            adapter.retrieve(state.query, a), adapter.retrieve(state.query, b),
        )
        self.assertEqual(first.evidence[0].passage_id, "a")
        self.assertEqual(second.evidence[0].passage_id, "b")
        self.assertEqual(a.bm25_weight, 1)

    async def test_original_request_and_output_contracts_still_work(self):
        from retrieval.schemas import RetrievalRequest

        adapter, _, _ = retrieval_service()
        items = adapter.pipeline.retrieve(RetrievalRequest(query="license", top_k=2))
        self.assertIsInstance(items[0], Evidence)


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    def controller(self, **kwargs):
        adapter, _, _ = retrieval_service(reverse=True)
        return OptimizationController(
            retrieval=adapter, generator=kwargs.pop("generator", Generator()),
            evaluator=kwargs.pop("evaluator", Evaluator()),
            rewriter=kwargs.pop("rewriter", Rewriter()), **kwargs,
        )

    async def test_regeneration_receives_revision_and_reuses_evidence(self):
        gen = Generator()
        state = make_input()
        result = await self.controller(generator=gen).run(state)
        self.assertEqual(result.stop_reason, "evaluation_passed")
        self.assertEqual((result.retrieval_rounds, result.evaluations, result.retry_count), (0, 1, 1))
        self.assertIsNotNone(gen.requests[0].revision)
        self.assertEqual(gen.requests[0].previous_response.answer, "Bring identification")
        self.assertEqual(state.retry_count, 0)

    async def test_query_branch_preserves_original_dialogue(self):
        state = make_input(feedback=feedback(failure_types=["query_misunderstanding"]))
        result = await self.controller().run(state)
        self.assertEqual(result.stop_reason, "evaluation_passed")
        self.assertNotEqual(result.query.rewritten_query, state.query.rewritten_query)
        self.assertEqual(result.query.current_message, state.query.current_message)
        self.assertEqual(result.query.domain, state.query.domain)

    async def test_hybrid_branch_executes_new_quotas(self):
        state = make_input(feedback=feedback(
            failure_types=["retrieval_balance"], preferred_retriever="bm25",
        ))
        result = await self.controller().run(state)
        self.assertEqual(result.stop_reason, "evaluation_passed")
        self.assertEqual(result.options.bm25_weight, 2)
        self.assertGreater(result.options.dense_top_k, state.options.dense_top_k)

    async def test_rerank_branch_accepts_same_ids_in_changed_order(self):
        adapter, dense, _ = retrieval_service(reverse=True)
        state = make_input()
        snapshot = await adapter.retrieve(state.query, state.options)
        state = make_input(
            evidence=snapshot.evidence, snapshot=snapshot,
            response=GeneratedResponse(answer="old", citations=["a"]),
            feedback=feedback(failure_types=["evidence_ranking"]),
            capabilities=adapter.capabilities,
        )
        result = await OptimizationController(
            retrieval=adapter, generator=Generator(), evaluator=Evaluator(), rewriter=Rewriter(),
        ).run(state)
        self.assertEqual(result.stop_reason, "evaluation_passed")
        self.assertEqual(result.retrieval_rounds, 0)
        self.assertEqual(result.evidence[0].passage_id, "b")
        self.assertEqual(len(dense.calls), 1)

    async def test_pass_does_not_generate(self):
        gen = Generator()
        result = await self.controller(generator=gen).run(make_input(feedback=feedback(decision="pass")))
        self.assertEqual(result.retry_count, 0)
        self.assertFalse(gen.requests)

    async def test_invalid_generated_citation_is_never_returned(self):
        result = await self.controller(generator=Generator(invalid=True)).run(make_input())
        self.assertEqual(result.response.status.value, "error")
        self.assertEqual(result.stop_reason, "invalid_citations")
        self.assertEqual(result.response.citations, [])

    async def test_retries_are_bounded_and_draft_is_not_published(self):
        class ChangingGenerator(Generator):
            async def generate(self, request, budget):
                response = await super().generate(request, budget)
                response.answer += f" Attempt {len(self.requests)}"
                return response

        result = await self.controller(
            generator=ChangingGenerator(), evaluator=Evaluator(retry=True),
        ).run(make_input())
        self.assertEqual(result.stop_reason, "retry_limit")
        self.assertEqual(result.retry_count, 2)
        self.assertEqual(len(result.candidate_history), 3)
        self.assertEqual(result.response.status.value, "escalated")
        self.assertFalse(result.escalation["delivered_to_human"])

    async def test_repeated_draft_and_feedback_stop_early(self):
        result = await self.controller(
            generator=Generator(same=True), evaluator=Evaluator(retry=True),
        ).run(make_input())
        self.assertEqual(result.stop_reason, "no_improvement")

    async def test_initial_and_consumed_budgets_are_respected(self):
        for budget, reason in (
            (BudgetState(used_tokens=None), "token_usage_unavailable"),
            (BudgetState(used_tokens=20001), "token_budget"),
            (BudgetState(elapsed_seconds=181), "latency_budget"),
        ):
            result = await self.controller().run(make_input(budget=budget))
            self.assertEqual(result.stop_reason, reason)

    async def test_deadline_cancels_awaited_component(self):
        result = await self.controller(evaluator=Evaluator(delay=0.1)).run(make_input(
            limits=Limits(max_latency_seconds=0.02),
        ))
        self.assertEqual(result.stop_reason, "latency_budget")

    async def test_clarify_and_escalation_disabled(self):
        result = await self.controller().run(make_input(feedback=feedback(decision="clarify")))
        self.assertEqual(result.response.status.value, "clarification_needed")
        result = await self.controller().run(make_input(
            feedback=feedback(decision="escalate"), limits=Limits(escalation_enabled=False),
        ))
        self.assertEqual(result.response.status.value, "insufficient_evidence")


class LLMAdapterAndBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_generator_prompt_contains_revision_and_maps_ids(self):
        llm = RecordingLLM({"answer": "fixed", "citations": ["E1"], "status": "answered"})
        request = GenerationRequest(
            query=make_input().query, evidence=[evidence()],
            previous_response=make_input().response, feedback=feedback(),
            revision=Revision(instruction="Add the missing deadline"),
        )
        budget = BudgetLedger(Limits(), BudgetState(used_tokens=100))
        result = await LLMRevisionGenerator(llm).generate(request, budget)
        self.assertEqual(result.citations, ["p::1"])
        self.assertIn("Add the missing deadline", llm.calls[0][0])
        self.assertEqual(budget.used_tokens, 108)

    async def test_rewriter_cannot_overwrite_user_question(self):
        llm = RecordingLLM({"rewritten_query": "fixed query", "current_message": "injected"})
        state = make_input()
        request = RewriteRequest(
            query=state.query, evidence=state.evidence, previous_response=state.response,
            feedback=state.feedback, options=state.options,
        )
        update = await LLMQueryRewriter(llm).rewrite(request, BudgetLedger(Limits(), BudgetState()))
        self.assertEqual(update.rewritten_query, "fixed query")
        self.assertNotIn("current_message", update.model_dump())

    async def test_main_evaluator_receives_matching_citations(self):
        llm = RecordingLLM(feedback(decision="pass").model_dump(exclude={
            "failure_types", "evidence_sufficient", "preferred_retriever",
        }))
        state = make_input()
        result = await MainEvaluatorAdapter(llm).evaluate(EvaluationRequest(
            query=state.query, evidence=state.evidence, response=state.response,
        ), BudgetLedger(Limits(), BudgetState()))
        self.assertEqual(result.decision.value, "pass")
        payload = llm.calls[0][2]
        self.assertEqual(payload["response"]["citations"][0],
                         payload["passages"][0]["doc_id"] + ":" + payload["passages"][0]["passage_id"])

    async def test_reservation_rejects_call_before_provider_runs(self):
        llm = RecordingLLM({})
        budget = BudgetLedger(Limits(max_total_tokens=10), BudgetState())
        with self.assertRaisesRegex(BudgetExceeded, "token_budget"):
            await budget.complete(llm, "hello", task="generate", payload={})
        self.assertEqual(llm.calls, [])

    async def test_unknown_provider_usage_stops_instead_of_zero(self):
        llm = RecordingLLM({}, tokens=None)
        budget = BudgetLedger(Limits(), BudgetState())
        with self.assertRaisesRegex(BudgetExceeded, "token_usage_unavailable"):
            await budget.complete(llm, "hello", task="generate", payload={})
        self.assertIsNone(budget.used_tokens)

    async def test_provider_error_marks_usage_unknown(self):
        budget = BudgetLedger(Limits(), BudgetState())
        with self.assertRaises(ValueError):
            await budget.complete(RecordingLLM(fail=True), "x", task="generate", payload={})
        self.assertIsNone(budget.used_tokens)

    async def test_generator_maps_exact_main_mock_citation(self):
        # Current Part 3 consumes native Evidence and accepts its legacy mock alias.
        item = evidence("CaseSensitive::1")
        llm = RecordingLLM({"answer": "fixed", "citations": [f"{item.doc_id}:{item.passage_id}"], "status": "answered"})
        state = make_input()
        request = GenerationRequest(
            query=state.query, evidence=[evidence("CaseSensitive::1")],
            previous_response=state.response, feedback=state.feedback,
        )
        result = await LLMRevisionGenerator(llm).generate(
            request, BudgetLedger(Limits(), BudgetState()),
        )
        self.assertEqual(result.citations, ["CaseSensitive::1"])

    async def test_call_costs_accumulate_across_components(self):
        ledger = BudgetLedger(Limits(max_total_tokens=2000), BudgetState(used_tokens=5))
        await ledger.complete(RecordingLLM({}, tokens=900), "x", task="plan", payload={})
        self.assertEqual(ledger.used_tokens, 906)
        await ledger.complete(RecordingLLM({}, tokens=900), "x", task="generate", payload={})
        self.assertEqual(ledger.used_tokens, 1807)
        llm = RecordingLLM({})
        with self.assertRaisesRegex(BudgetExceeded, "token_budget"):
            await ledger.complete(llm, "x", task="evaluate", payload={})
        self.assertFalse(llm.calls)


class StopConditionTests(unittest.IsolatedAsyncioTestCase):
    async def test_unchanged_query_is_rejected_before_retrieval(self):
        class SameRewriter:
            async def rewrite(self, request, budget):
                return QueryUpdate(rewritten_query=request.query.rewritten_query,
                                   sub_queries=request.query.sub_queries)

        adapter, dense, _ = retrieval_service()
        result = await OptimizationController(
            retrieval=adapter, generator=Generator(), evaluator=Evaluator(),
            rewriter=SameRewriter(),
        ).run(make_input(feedback=feedback(failure_types=["query_misunderstanding"])))
        self.assertEqual(result.stop_reason, "no_effective_action")
        self.assertEqual(dense.calls, [])

    async def test_unchanged_context_stops_without_publishing_draft(self):
        adapter, _, _ = retrieval_service()
        state = make_input()
        snapshot = await adapter.retrieve(state.query, state.options)
        # Only a,b are available, so larger quotas/context cannot add evidence.
        adapter.pipeline.dense_retriever.ids = ["a", "b"]
        adapter.pipeline.bm25_retriever.ids = ["b", "a"]
        generator = Generator()
        result = await OptimizationController(
            retrieval=adapter, generator=generator, evaluator=Evaluator(),
        ).run(make_input(
            evidence=snapshot.evidence, snapshot=snapshot,
            response=GeneratedResponse(answer="rejected draft", citations=["a"]),
            feedback=feedback(failure_types=["retrieval_coverage"], evidence_sufficient=False),
        ))
        self.assertEqual(result.stop_reason, "no_context_change")
        self.assertFalse(generator.requests)
        self.assertNotEqual(result.response.answer, "rejected draft")

    async def test_empty_evidence_is_bounded_without_generation(self):
        adapter, dense, bm25 = retrieval_service()
        dense.ids, bm25.ids = [], []
        generator = Generator()
        result = await OptimizationController(
            retrieval=adapter, generator=generator, evaluator=Evaluator(),
        ).run(make_input(evidence=[], response=GeneratedResponse(
            answer="insufficient", citations=[], status="insufficient_evidence",
        )))
        self.assertEqual(result.stop_reason, "retry_limit")
        self.assertEqual(result.retry_count, 2)
        self.assertFalse(generator.requests)

    async def test_missing_real_reranker_does_not_silently_noop(self):
        adapter, _, _ = retrieval_service()
        state = make_input(capabilities=Capabilities(reranking=True),
                           feedback=feedback(failure_types=["evidence_ranking"]))
        result = await OptimizationController(
            retrieval=adapter, generator=Generator(), evaluator=Evaluator(),
        ).run(state)
        self.assertEqual(result.stop_reason, "unsupported_branch:rerank")

    async def test_component_error_returns_error_not_draft(self):
        class BrokenGenerator:
            async def generate(self, request, budget):
                raise RuntimeError("broken")

        adapter, _, _ = retrieval_service()
        result = await OptimizationController(
            retrieval=adapter, generator=BrokenGenerator(), evaluator=Evaluator(),
        ).run(make_input())
        self.assertEqual(result.stop_reason, "component_error:RuntimeError")
        self.assertEqual(result.response.status.value, "error")

    async def test_non_finite_custom_reranker_is_rejected(self):
        class BadReranker:
            def rerank(self, query, sub_queries, candidates):
                candidates[0].rerank_score = float("nan")
                return candidates

        adapter, _, _ = retrieval_service()
        adapter.rerankers["identity"] = BadReranker()
        state = make_input()
        with self.assertRaisesRegex(ValueError, "non-finite"):
            await adapter.retrieve(state.query, ExecutionOptions(**{
                **state.options.model_dump(), "rerank": True, "reranker_query_strategy": "identity",
            }))


if __name__ == "__main__":
    unittest.main()
