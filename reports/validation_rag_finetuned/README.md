# MultiDoc2Dial validation: no planner and no domain filter

This experiment processes the 4,427 official validation requests. It imports
existing code from `main/` without changing it and reads the previously built
struct fine-tuned index without changing that index.

- No LLM question rewrite, domain prediction, or subqueries.
- Retrieval input uses `main`'s `finetuned_query`: the current user utterance
  and at most three preceding visible turns, with User/Agent labels. This is a
  deterministic formatting step used by the fine-tuned model, not a generated
  rewrite. The retrieval request sets `domain=None`.
- Main's struct fine-tuned BGE Dense pipeline retrieves from the full corpus.
  The final context is Top 10, with no BM25 or reranker.
- Main's `EvidenceResponseGenerator` produces a cited DeepSeek answer.
- Reference answers and gold evidence are used only after generation to score
  Token F1, Exact Match, evidence recall and citations.

`results/<run>/events.jsonl` saves every phase as it completes. `cases.jsonl`
contains the query, top 100 Dense ranking, top 50 candidates, final evidence,
generation requests, raw model responses, usage, and errors.
`evaluation.jsonl` and `metrics.json` hold per-request and aggregate scores.
After all 4,427 requests, `comparison.json` pairs the scores with the earlier
domain-routed run on the same validation samples.

The DeepSeek key is supplied only to the running process and is not written to
the experiment files. Validation is a development split; it does not prove
cross-domain performance or independent factual correctness.

## Completed run

`results/validation_20261109/` contains 4,427 unique cases, 13,281 phase
events, and 4,427 per-case evaluations. Every case used no domain filter, no
subqueries or planner calls, and 10 evidence passages.

| Measure | Result |
| --- | ---: |
| Answer Token F1 | 0.2122 |
| Answer Exact Match | 0.00068 |
| Gold evidence recall at 10 | 0.7944 |
| Gold evidence hit rate at 10 | 0.7965 |
| Citation evidence validity | 1.0000 |
| Citation gold precision | 0.3696 |

DeepSeek returned 4,263 answered, 139 insufficient-evidence, and 25 error
statuses. All 25 errors were validation failures when an insufficient-evidence
response also cited evidence; their raw responses and errors are retained.
