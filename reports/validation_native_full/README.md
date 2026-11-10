# Official validation: full pipeline with struct fine-tuned retrieval

This report scores all 4,427 official MultiDoc2Dial validation requests. The local run used `configs/native.yaml`: DeepSeek Planner, deterministic last-four-visible-turn first retrieval query, struct fine-tuned BGE Dense retrieval across all domains, Top 10 evidence, DeepSeek answer generation, online evaluation, and up to two optimization retries. There is no BM25 or Cross-Encoder reranker. The same `sample_id` set and validation file hash are used as in the team's RAG + generation baseline.

Two requests ended in `error` after the generator received an `insufficient_evidence` response with citations, which violates the output contract. The latest saved attempt for each was retained. Both remain in the **4,427-request denominator** and receive zero answer F1/EM, matching the baseline's treatment of its 25 format errors. No error was relabeled as clarification.

| Metric | RAG + generation baseline | Full pipeline |
| --- | ---: | ---: |
| Answer Token F1 | 0.2122 | 0.1837 |
| Answer Exact Match | 0.00068 | 0.00068 |
| Gold evidence recall at 10 | 0.7944 | 0.7967 |
| Gold evidence hit rate at 10 | 0.7965 | 0.7987 |
| Citation evidence validity | 1.0000 | 1.0000 |
| Citation gold precision | 0.3696 | 0.3754 |

Full-pipeline final statuses: 3,639 answered, 580 escalated, 206 clarification needed, 2 error. Mean per-request latency is 6.46 seconds; mean recorded model calls 3.39 and mean total tokens 7,358.4. The baseline's uploaded aggregate does not include the same latency and token fields, so no cost ratio is reported here.

The full pipeline's answer F1 is **2.85 percentage points lower** than the RAG + generation baseline, while evidence recall at 10 is **0.23 percentage points higher**. This is an observed difference between workflows, not evidence that the optimizer alone caused the change: the full pipeline also invokes a Planner and may choose clarification or escalation. Non-answered outcomes score zero against official answers. Validation is a development split, and citation validity does not prove factual correctness.

Files: `metrics.json` holds aggregate scores; `evaluation.jsonl` holds one compact score row per request; `manifest.json` identifies the inputs and error policy. The raw model traces, API key, downloaded data, checkpoint, and index stay outside this report. With the official processed validation file, checkpoint, index, and DeepSeek key available locally, run `python -m experiments.validation.full --output results/validation_native_full` from the repository root. The command resumes completed requests. Then regenerate this compact report with `python -m experiments.validation.report`. Set `PYTHONPATH=src` for both commands when the package is not installed in editable mode.
