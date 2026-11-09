# Evidence map

All paths below are relative to `HKUST-26fall-nlp-` unless otherwise stated. Historical outputs are analyzed as saved; the paper does not assert a rerun of the current refactored code. No new model run was performed.

| Paper content | Saved source |
|---|---|
| Native segmentation, dense/hybrid/reranking comparison | `reports/project_report/retrievalreport_en.docx` (unchanged supplied input) |
| Current fine-tuned preset and execution bypass | `configs/native.yaml`; `src/customer_service_rag/retrieval/config.py` and `pipeline.py`; `src/customer_service_rag/optimization/retrieval_adapter.py` |
| Distinct retained hybrid configuration | `configs/deepseek.yaml`; the historical per-run configs/manifests listed below |
| Adaptation hyperparameters and checkpoint selection | `models/bge-base-en-v1.5-struct-ft/experiment/training_config.json`, `training_summary.json`, `selection.json` |
| 400-query adaptation comparison | `models/bge-base-en-v1.5-struct-ft/experiment/final_comparison.json` |
| Official-validation single pass (1,000 requests) | `reports/validation_a_seed42_n1000/metrics.json`, `summary.csv`, `predictions.jsonl`, `selection.json`, `config.yaml` |
| Cross-domain initial A/B/C | `reports/cross_domain/20261107T194151Z/` (`manifest.json`, `main_summary.json`, `cases.jsonl`, `point_grades.jsonl`, `case_summary.csv`) |
| Cross-domain evaluation/optimization continuation | `reports/cross_domain/20261108_full/` (same file types) |
| Synthetic requests and required points | `data/cross_domain/cross_domain_20_annotations.json` |
| Conditional error (018), internal status/citation error (009), local synthesis gain (017), retry-to-pass example (010) | Corresponding records in the two cross-domain `cases.jsonl` files |
| Identical text with changed point labels (005/014/018) | Compare answer text in `cases.jsonl` and labels in `point_grades.jsonl` across the two cross-domain batches |
| Dummy-file disclaimer | `data/raw/multidoc2dial/multidoc2dial/README.md`; official dataset data README |

The exploratory paired intervals were recomputed offline from the recorded 20 per-question outcomes: 50,000 paired bootstrap resamples, seed 42, percentile 95% intervals. Resampling preserves the paired A/B/C identity of each request. It does not account for synthetic-label or judge uncertainty. The R2–R4 dialogue-cluster interval comes from the retrieval artifact rather than treating its 400 requests as independent dialogues.

Figures use the same aggregate values as the paper tables. Coverage is a macro-average across requests; non-answered final outputs count as zero. Tokens are input plus output tokens and include generation/evaluation within each recorded arm's workflow. Shared planning and offline point grading are excluded. No end-to-end latency comparison is inferred from continuation times.
