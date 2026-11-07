# Validation A：规划、检索到 Generator

这批结果使用官方 MultiDoc2Dial `validation` 划分，在 4,427 条预处理请求中用 Python `random.Random(42).sample` 固定抽取 1,000 条。在线流程为 Query Planner → Dense + BM25 → RRF → Cross-Encoder Reranker → Response Generator；质量评价、优化循环和转接均关闭。这是 A 的单次运行结果，不是完整流程 B，也不是 Dense-only Single-RAG 基线。

## 可向小组同步的进度

> A 的真实模型联调已在官方 validation 随机 1,000 条上完成，代码基于 `main` 的 `ccf5933`，模型为 DeepSeek `deepseek-flash`，检索使用真实 MiniLM Embedding 与 Cross-Encoder Reranker。973 条正常回答、25 条返回证据不足、2 条因生成输出格式不符合契约而报错。回答 Token F1 为 0.2080，EM 为 0；Evidence Recall@5 为 0.6813。结果可按固定抽样清单和配置复核。B 的 Evaluator–Optimizer 完整流程尚未运行。

## 配置与指标

| 项目 | 结果 |
| --- | ---: |
| 官方 validation 总请求数 / 抽样数 | 4,427 / 1,000 |
| 随机种子 | 42 |
| `answered` / `insufficient_evidence` / `error` | 973 / 25 / 2 |
| Token F1 / EM（全 1,000 条） | 0.2080 / 0 |
| Evidence Recall@1 / @5 / @10 | 0.4000 / 0.6813 / 0.7703 |
| 引用 ID 在所给证据中的有效率 | 1.0000 |
| 引用 Gold precision | 0.3222 |
| 平均 / P95 单请求耗时 | 4.73 / 6.05 秒 |
| 平均模型调用 / Token | 2.02 次 / 2,127 |

F1/EM 对有参考答案但未作答或报错的样本按 0 计。Evidence Recall 按必要证据组计算；引用 ID 有效不代表回答事实正确。未做独立事实标注，因此幻觉率、grounding accuracy、对话一致性和真实任务完成率未报告。耗时是单请求运行耗时，不含模型加载与建索引。

两条 `error` 均是模型在 `insufficient_evidence` 状态下仍返回引用，内置格式纠正后仍违反契约。运行过程也遇到过 DeepSeek HTTP 500；等待后重试恢复，最终这两条错误与 HTTP 500 无关。保留错误状态，不把它们改写为正常回答。

## 文件与复现

- [`selection.json`](selection.json)：固定抽样的 1,000 个 `sample_id`。
- [`config.yaml`](config.yaml)：展开后的实际 A 配置，无密钥。
- [`predictions.jsonl`](predictions.jsonl)：每条样本的最终状态、回答、引用和 `run_id`。
- [`summary.csv`](summary.csv)：逐样本 F1、EM 和证据召回。
- [`metrics.json`](metrics.json)：汇总指标及未测量字段。

本地完整原始轨迹保存在被 `.gitignore` 排除的 `results/<run_id>/`，未上传 GitHub。使用 `scripts/run_validation_a_sample.py --count 1000 --seed 42` 可在官方数据、索引、模型及本地 DeepSeek 密钥就绪后复跑；`scripts/finalize_validation_a_sample.py` 从本地逐条轨迹生成轻量结果。原始运行使用的项目提交为 `ccf5933`；本目录和运行脚本是其后添加的结果交付文件。
