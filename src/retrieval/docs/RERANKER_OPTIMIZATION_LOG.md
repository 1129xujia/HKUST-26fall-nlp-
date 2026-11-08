# Reranker Optimization Log

本文档记录 Part 2 Cross-Encoder 重排优化过程。召回阶段固定使用现有 BGE、BM25 与 RRF，不把召回变化误记为重排收益。所有实验同时保留正结果、负结果和未验证猜想。

## 2026-10-29：起点与目标

### 固定的召回条件

- Dense：`BAAI/bge-base-en-v1.5`
- BM25：查询侧移除 `User` / `Agent` 角色标记
- RRF：`rrf_k=30`
- 额外语义/每文档去重：关闭
- Cross-Encoder：`cross-encoder/ms-marco-MiniLM-L6-v2`
- validation：80 条按领域均衡样本，domain filter 开启

### 已知召回上限

两路各取 Top 200 后进行 RRF，重排前候选覆盖为：

| 候选深度 | Hit Rate | Recall |
| --- | ---: | ---: |
| 50 | 61.25% | 58.13% |
| 100 | 67.50% | 63.12% |
| 200 | 75.00% | 72.50% |

这些值是对应候选深度下重排的理论上限。候选池没有 Gold 的样本不能归责于 Cross-Encoder。

### 第一轮计划

1. 缓存相同的 RRF Top 200 候选，避免不同重排策略看到不同输入。
2. 比较候选深度 50/100/200。
3. 比较 history、current、user history、current-first user history 四种重排 query；召回 query 始终保持 history。
4. 对每条样本记录 Gold 重排前/后最佳名次，统计 promotion、demotion、保持不变及条件 Hit@10。
5. 第一轮不训练模型；先判断失败来自候选缺失、query 格式还是模型领域能力。

### 工程调整：分阶段运行

首次尝试一次运行四种 query、Top 200、`batch_size=32`。在当前 16 GB Mac CPU 环境中，batch 32 因内存压力只有约 1–2 batch/s，反而慢于 batch 16；长任务还会在完成前失去全部未落盘结果，因此手动中止，没有把不完整输出计入实验。

调整为两阶段：先让四种 query 在相同 Top 100 候选上独立运行并各自落盘，用于选择 query；再只让表现最好的 query 扩展到 Top 200。batch 固定为 16。这样不改变实验目标，同时避免把计算资源浪费在已明显落后的 query 上。

## 实验 1：重排 Query 消融

召回始终使用 history；下表只改变 Cross-Encoder 输入。为公平比较，四种策略共享两路 Top 100 融合后的同一批 RRF 候选，并在其中取前 50 条重排。

| Rerank query | Hit@10 | Recall@10 | MRR@10 | Gold 提升率 | Gold 压低率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 完整 history | 37.50% | 33.12% | 0.2242 | 20.41% | 42.86% |
| 仅当前问题 | 46.25% | 41.88% | 0.3115 | 38.78% | 38.78% |
| 仅 user history | 48.75% | 43.13% | 0.2987 | 38.78% | 36.73% |
| 当前问题置前 + 之前 user history | **55.00%** | **48.75%** | **0.3418** | **42.86%** | **28.57%** |

同一候选池未重排的 RRF Hit@10/Recall@10 为 47.50%/43.33%。完整 history 明显破坏排序；只用当前问题虽然快，但会丢失指代上下文。最佳策略把当前问题放在开头，同时保留之前的用户问题，不包含 agent 长回答。

采取的决策：召回 query 不变；`full` 模式只在 Cross-Encoder 前转换 main query。子查询仍用于扩大召回，但在缺少专门消融证据前，不参与 Cross-Encoder 的 `max` 打分，避免错误子查询支配排序。

## 实验 2：候选深度 50/100/200

使用最佳 query，对两路各 Top 200 的同一融合列表改变进入 Cross-Encoder 的候选数量：

| 重排深度 | 候选 Hit 上限 | 重排后 Hit@10 | Recall@10 | MRR@10 | CPU 打分均值/样本 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50 | 61.25% | 55.00% | 48.75% | 0.3370 | 约 1.1 s（按规模与单独 Top 50 实测） |
| 100 | 67.50% | 52.50% | 46.88% | 0.3437 | 约 2.3 s |
| 200 | 75.00% | **56.25%** | **50.62%** | **0.3511** | 4.38 s |

Top 200 绝对指标最高，但相对 Top 50 只增加 1.25 pp Hit 和 1.87 pp Recall，CPU 延迟约为四倍。Top 100 还出现指标回落，说明候选越多并不保证最终 Top 10 越好。

采取的决策：线上 `full` 模式选择融合 Top 50。Top 200 保留为离线高召回实验，不作为当前交互路径默认值。

## 实验 3：Passage 输入格式

固定最佳 query、相同候选列表，比较：

- `index_text`：文档标题、父标题、section 标题、正文；
- `labeled`：显式 `Document/Section/Content` 标签，不包含父标题链；
- `text_only`：仅正文。

在两路各 Top 50 的严格同候选实验中：

| Passage 格式 | Hit@10 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: |
| index_text | 53.75% | 48.13% | 0.3407 |
| labeled | 53.75% | 48.75% | 0.3431 |
| text_only | 48.75% | 41.04% | 0.2576 |

差异很小，因此又在最终线上 Dense 100 / BM25 50 / RRF Top 50 配置复测：`index_text` 为 55.00%/48.75%，`labeled` 为 53.75%/48.13%。

采取的决策：继续使用 `Passage.index_text`。标题和层级信息对政府服务 section 很重要；仅正文被明确否定。

## 实验 4：开发样本与留出样本

### 第一批 80 条，线上精确配置

Dense 100、BM25 50、RRF Top 50、最佳 query、`index_text`：

| 指标 | 重排前 RRF | 重排后 | 变化 |
| --- | ---: | ---: | ---: |
| Hit@5 | 42.50% | 47.50% | +5.00 pp |
| Recall@5 | 36.67% | 40.62% | +3.96 pp |
| Hit@10 | 48.75% | 55.00% | +6.25 pp |
| Recall@10 | 44.58% | 48.75% | +4.17 pp |
| MRR@10 | 0.3090 | 0.3476 | +0.0387 |

候选包含 Gold 的 49 条样本中，44.90% 的最佳 Gold 排名被提升，28.57% 被压低；7 条被救入 Top 10，2 条从 Top 10 掉出。

### 下一批 80 条留出样本

| 指标 | 重排前 RRF | 重排后 | 变化 |
| --- | ---: | ---: | ---: |
| Hit@10 | 66.25% | 67.50% | +1.25 pp |
| Recall@10 | 61.88% | 64.38% | +2.50 pp |
| MRR@10 | 0.3618 | 0.4710 | +0.1092 |

留出集提升幅度不同，但 Hit、Recall、MRR 三项方向均为正，因此 query 策略不是只在第一批 80 条上成立。

## 正式 Pipeline 复核

- 第一批输出：`artifacts/evaluation_full_reranker_optimized_80.json`
- 留出输出：`artifacts/evaluation_full_reranker_optimized_holdout_80.json`
- 第一批真实平均延迟：1,428 ms，P95 3,052 ms；RRF 约 102 ms。
- 留出集真实平均延迟：1,218 ms，P95 2,008 ms。
- 单元与集成测试：56 passed。

真实 `HybridRetrievalPipeline.from_artifacts` 与离线实验指标完全一致，说明提升已经进入正式调用链，而不是只存在于实验脚本。

## 当前结论与微调决策

当前通用 MiniLM Cross-Encoder 在修正 query 表示后已经产生可复现净收益，因此“必须立刻微调模型”的假设暂不成立。此时直接微调会同时引入训练数据构造、false negative 和文档泄漏风险。

当前默认策略：

```text
召回 query        = 原 history + sub_queries
Dense Top K       = 100
BM25 Top K        = 50
RRF k             = 30
Rerank candidates = 50
Rerank query      = 当前 user 问题置前 + 之前 user 问题
Rerank sub-query  = 暂不参与打分
Passage input     = index_text
Extra dedup       = off
```

下一阶段若继续优化，优先处理 CPU 延迟和被压低样本审计；只有在确认剩余错误主要来自领域判别能力后，再使用 dialogue/document 隔离的数据划分和 RRF hard negatives 微调。

## 实验 5：低信息确认回答保护

对第一批中“重排前在 Top 10、重排后掉出”的 2 条样本审计：

1. `yes it is`：Gold 从 RRF Rank 1 被压到 Rank 34；
2. `I do.`：Gold 从 Rank 5 被压到 Rank 25。

两条都是对上一条 agent 澄清问题的确认，本身没有独立检索语义。完整 history、仅 current 和 user-history 都未能稳定救回，说明这类 query 不适合强制交给 Cross-Encoder。

比较了三个保护规则：短词数阈值、宽泛短句、显式确认词白名单。词数阈值会把 `what about my browser?` 等真实短问题一起跳过，因此拒绝采用。最终仅对白名单 `yes/no/yeah/yep/nope/yes it is/I do/...` 跳过模型并保留 RRF。

### 第一批 80 条真实 Pipeline

| 指标 | 无保护重排 | 确认词保护 | 变化 |
| --- | ---: | ---: | ---: |
| Hit@5 | 47.50% | 50.00% | +2.50 pp |
| Recall@5 | 40.62% | 43.75% | +3.13 pp |
| Hit@10 | 55.00% | **57.50%** | +2.50 pp |
| Recall@10 | 48.75% | **52.50%** | +3.75 pp |
| MRR@10 | 0.3476 | **0.3751** | +0.0275 |
| 平均延迟 | 1,428 ms | **1,219 ms** | -209 ms |

### 下一批 80 条留出样本

确认词保护不改变 Hit@10 67.50% 和 Recall@10 64.38%，MRR 从 0.4710 提高到 0.4800，平均延迟从 1,218 ms 降到 1,088 ms。说明保护策略在留出样本上没有牺牲覆盖率，并减少了无意义模型调用。

最终正式输出：

- `artifacts/evaluation_full_reranker_guarded_80.json`
- `artifacts/evaluation_full_reranker_guarded_holdout_80.json`
- 测试：57 passed。

最终 `full` 策略在前述配置基础上增加：确认词回答跳过 Cross-Encoder，沿用 RRF 顺序。微调仍暂缓，因为输入修正与安全保护已在两批样本上获得可复现净收益；下一轮微调应针对剩余 demotion 样本，而不是盲目训练全部数据。

## 最终端到端对照（线上精确口径）

下表全部由 `scripts.evaluate_retrieval` 经过正式 Pipeline 产生，Dense 100、BM25 50，其余条件和样本完全一致：

| 样本 | 模式 | Hit@10 | Recall@10 | MRR@10 | 平均延迟 |
| --- | --- | ---: | ---: | ---: | ---: |
| 第一批 80 | RRF | 48.75% | 44.58% | 0.3090 | 101.5 ms |
| 第一批 80 | 最终 guarded Full | **57.50%** | **52.50%** | **0.3751** | 1,219 ms |
| 留出 80 | RRF | 67.50% | 62.50% | 0.3630 | 111.1 ms |
| 留出 80 | 最终 guarded Full | **67.50%** | **64.38%** | **0.4800** | 1,088 ms |

第一批的 Hit/Recall 明显提高；留出集 Hit 持平、Recall 和 MRR 提高，没有出现重排导致整体退化。代价是 CPU 延迟约增加到 10–12 倍，所以默认命令仍为 `rrf`，`full` 由调用方在更看重排序质量时显式启用。

最终对照文件：

- `artifacts/evaluation_rrf_optimized_roles_80.json`
- `artifacts/evaluation_full_reranker_guarded_80.json`
- `artifacts/evaluation_rrf_optimized_holdout_80.json`
- `artifacts/evaluation_full_reranker_guarded_holdout_80.json`

## 扩大 Validation：四领域均衡 400 条

为降低 80 条实验的随机波动，使用 `--balanced-by-domain --limit 400` 扩大评估。样本来自同一 validation 集，`dmv`、`ssa`、`studentaid`、`va` 各 100 条；RRF 与 guarded Full 使用完全相同的样本顺序和 Gold，唯一主要差异是是否启用 Cross-Encoder 重排。

本次 400 条占全部 4,496 条可评估 validation 样本的约 8.9%，属于扩大抽样验证，不等同于完整 validation。

| 指标 | RRF | guarded Full | 变化 |
| --- | ---: | ---: | ---: |
| Hit@5 | 43.50% | **52.25%** | **+8.75 pp** |
| Recall@5 | 36.46% | **45.88%** | **+9.42 pp** |
| Hit@10 | 55.50% | **60.75%** | **+5.25 pp** |
| Recall@10 | 49.42% | **55.38%** | **+5.96 pp** |
| MRR | 0.3084 | **0.3997** | **+0.0912** |
| 平均延迟 | **116.0 ms** | 1,187.9 ms | +1,071.9 ms |
| P50 延迟 | **108.8 ms** | 1,065.9 ms | +957.1 ms |
| P95 延迟 | **197.4 ms** | 2,315.8 ms | +2,118.4 ms |

### User / Agent Gold 拆分

| 指标 | RRF | guarded Full | 变化 |
| --- | ---: | ---: | ---: |
| User Gold Hit@5 | 38.75% | **43.75%** | +5.00 pp |
| User Gold Recall@5 | 38.75% | **43.63%** | +4.88 pp |
| Agent Gold Hit@5 | 34.18% | **48.72%** | +14.54 pp |
| Agent Gold Recall@5 | 34.18% | **48.60%** | +14.41 pp |
| User Gold Hit@10 | 52.00% | **54.75%** | +2.75 pp |
| User Gold Recall@10 | 52.00% | **54.63%** | +2.63 pp |
| Agent Gold Hit@10 | 46.94% | **56.89%** | +9.95 pp |
| Agent Gold Recall@10 | 46.81% | **56.76%** | +9.95 pp |

扩大样本后，重排在总体 Hit、Recall 和 MRR 上仍为稳定正收益，并且 Agent Gold 的提升显著大于 User Gold。这说明经过 query 表示修正和确认词保护后，当前 Cross-Encoder 尤其擅长把与后续 agent 回答相关的证据推到前列；User Gold 也有提升，但 Top 10 的增益较小，后续错误审计应优先区分“候选池没有 User Gold”和“User Gold 已召回但重排不足”。

代价仍然明显：guarded Full 的平均延迟约为 RRF 的 10.2 倍。因此当前结论维持不变：质量优先时显式使用 `full`，延迟敏感时使用默认 `rrf`。下一步若需要更强统计结论，可以继续运行全部 4,496 条 validation，但按本机当前速度，Full 预计需要约 1.5 小时。

本次输出：

- `artifacts/evaluation_rrf_balanced_400.json`
- `artifacts/evaluation_full_reranker_guarded_balanced_400.json`
