# Retrieval Optimization Log

召回优化后的 Cross-Encoder 专项实验已转入 `RERANKER_OPTIMIZATION_LOG.md`，本文继续保留召回阶段的完整决策链。

本文档记录 Part 2 检索优化过程，供实验报告、项目复盘和后续组员交接使用。所有结论都区分为“已观察事实”“待验证猜想”和“采取的决策”，避免把猜想写成结论。

## 2026-10-29：初始问题与基线

### 目标

优先解决 Dense、BM25 和 RRF 的候选召回及最终 Top-K 命中问题。在一阶段检索稳定前，暂不优化 Cross-Encoder reranker。

### 已观察事实

使用 80 条按领域均衡抽取的 validation 样本、4-turn history query、domain filter：

| 方法 | Recall@10 | Recall@30 | Recall@50 | Recall@100 | Recall@200 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 原始候选 | 38.54% | 48.96% | 55.00% | 60.00% | 65.63% |
| Dense 原始候选 | 37.50% | 52.50% | 55.00% | 61.88% | 71.25% |

| 方法 | Hit@10 | Hit@30 | Hit@50 | Hit@100 | Hit@200 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 原始候选 | 43.75% | 53.75% | 57.50% | 63.75% | 67.50% |
| Dense 原始候选 | 42.50% | 56.25% | 58.75% | 66.25% | 76.25% |

Top 200 时，两路都未命中任何 Gold 的样本为 15/80；理想两路并集 Hit Rate 为 81.25%，Recall 为 78.13%。这说明两路存在明显互补，同时仍有一批共同失败样本。

之前的 Pipeline 结果：RRF Hit@10 为 33.75%、Recall@10 为 27.29%；加入 Cross-Encoder 后下降到 Hit@10 23.75%、Recall@10 17.29%。

### 发现的问题

1. 单路 Top 30 截断过早：扩大到 Top 200 后 Recall 提升约 17–19 个百分点。
2. 原始候选进入 Pipeline 后损失明显，可能发生在 RRF 排序、候选截断或去重阶段。
3. Cross-Encoder 明确降低当前验证集指标。
4. 15/80 样本在两路 Top 200 内共同失败，无法只靠扩大候选池解决。
5. validation 评估当前没有传入 `sub_queries`，所以已有低指标不能归因于子查询。
6. User Gold 与 Agent Gold 的覆盖率接近，暂未发现某一类 Gold 是唯一瓶颈。
7. Dense 使用 `faiss.IndexFlatIP` 精确搜索，不存在近似 FAISS 索引漏搜问题。

### 提出的猜想

1. `EvidenceDeduplicator` 的每文档数量限制或语义相似阈值可能误删 Gold。
2. 直接拼接最近 4 个 turn 可能引入旧话题和 agent 文本噪声；独立问题改写可能优于 raw history。
3. Dense/BM25 的候选池过小，且 RRF 参数可能无法把后排互补结果提升到最终 Top 10。
4. `BAAI/bge-base-en-v1.5` 是通用模型，可能不充分适配多轮政府服务对话。
5. 当前 BM25 只有基础正则分词，没有词形归一、字段权重和领域缩写扩展。
6. section-level passage 可能过长，导致主题稀释和相邻证据被错误去重。

### 已采取的决策

1. 生产默认路径改为 Dense + BM25 + RRF，`use_reranker=False`；额外去重是否关闭由实验 1 决定。
2. `full` 模式继续保留，只用于后续显式重排消融。
3. Gold 拆分为 User/Agent 两组，并移除重复指标别名。
4. 新增原始候选 Recall@10/30/50/100/200 诊断及逐 Gold 排名记录。

## 优化执行顺序

1. 去重开关消融，确认 Pipeline 候选损失来源。
2. 比较 current、history 和可复现的 query rewrite。
3. 搜索 Dense/BM25 candidate Top-K。
4. 搜索 RRF `k`、融合候选上限及最终 Top-K 行为。
5. 在固定 query/corpus/参数后对比 embedding 模型。
6. 优化 BM25 分词、词形和字段权重。
7. 比较 section、span 和较小 chunk 的语料粒度。
8. 人工归类两路共同失败样本。
9. 一阶段检索稳定后再处理 reranker。

## 实验记录

### 实验 1：去重消融

- 状态：已完成
- 控制变量：相同 80 样本、history query、domain filter、Dense/BM25 Top 30、RRF k=60。
- 对比：当前语义去重 vs 仅按排序截取 Top 10。
- 判断标准：若关闭去重后 Hit/Recall 明显提高，则进一步拆分“每文档上限”和“语义相似过滤”的影响。

结果：

| 去重方式 | Hit@10 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: |
| 关闭全部去重 | 45.00% | 40.83% | 0.3052 |
| 仅标准化正文完全相同去重 | 41.25% | 37.08% | 0.2868 |
| 仅每文档最多两段 | 35.00% | 27.92% | 0.2749 |
| 仅 embedding 相似度去重 | 37.50% | 32.92% | 0.2720 |
| 当前完整去重 | 33.75% | 27.29% | 0.2648 |

结论：去重是当前 Pipeline 候选损失的主要来源。每文档最多两段的限制伤害最大，语义去重也会误删 Gold；即使标准化正文完全相同过滤也会降低指标，说明数据中的不同 passage ID 可能共享正文，但在 Gold 结构上仍不可互换。RRF 本身已经按 passage ID 合并同一候选，因此一阶段检索阶段暂时关闭额外去重。

采取的改动：增加显式去重开关，默认关闭；保留原实现供生成阶段或后续消融使用。

### 实验 2：Query 构造消融

- 状态：部分完成
- 对比：current utterance、4-turn history、独立问题改写。
- 注意：当前仓库没有生成式 query rewriter，必须明确区分真实 rewrite 与简单字符串拼接。

已完成结果（关闭去重，其余参数保持 Dense/BM25 Top 30、RRF k=60）：

| Query | Hit@10 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: |
| 最近 4 turn history | 45.00% | 40.83% | 0.3052 |
| 仅 current utterance | 40.00% | 36.46% | 0.2339 |

结论：当前样本中直接丢弃历史会降低整体指标，说明指代和上下文确实重要。该结果不能证明原始 history 已经最优；它只排除了“统一改成 current-only”这一方案。真实 standalone rewrite 尚未实现和测量，继续列为待办。

### 实验 3：候选池与 RRF 参数

- 状态：第一轮已完成
- 对比：candidate Top-K 30/50/100/200，RRF k 10/30/60/100。

使用缓存的单路 Top 200 排名，对 current/history、Dense K、BM25 K 和 RRF k 做网格搜索；关闭 reranker 与去重，最终取 Top 10。

基线：

```text
history + Dense 30 + BM25 30 + RRF k=60
Hit@10=45.00%，Recall@10=40.83%，MRR@10=0.3052
```

本轮最佳：

```text
history + Dense 100 + BM25 50 + RRF k=30
Hit@10=47.50%，Recall@10=43.33%，MRR@10=0.3058
```

相对关闭去重后的基线，Hit@10 和 Recall@10 均提升 2.5 个百分点。扩大到 200 并不自动得到最佳最终 Top 10，说明候选深度与 RRF 排名常数需要共同调节，而不是一味增大候选池。

采取的改动：默认 `dense_top_k=100`、`bm25_top_k=50`、`rrf_k=30`。保留参数化配置，后续在更大样本上验证稳定性。

### 当前阶段性最优配置

```text
query_mode = history（最近 4 turn）
Dense candidate Top K = 100
BM25 candidate Top K = 50
RRF k = 30
reranker = off
deduplication = off
final evaluation Top K = 10
```

80 条均衡样本的离线缓存模拟指标：Hit@10 47.50%、Recall@10 43.33%、MRR@10 0.3058。下一步需要用真实 Pipeline 复核，确保缓存模拟与生产调用链一致。

### 真实 Pipeline 复核

- 状态：已完成
- 输出：`artifacts/evaluation_rrf_optimized_80.json`
- 测试：53 passed（另有 1 条 LangGraph 依赖弃用警告，与本次修改无关）

真实 `HybridRetrievalPipeline.from_artifacts` 使用阶段性最优配置得到：

| 指标 | 原 RRF | 优化后 RRF | 变化 |
| --- | ---: | ---: | ---: |
| Hit@5 | 28.75% | 41.25% | +12.50 pp |
| Recall@5 | 22.92% | 35.42% | +12.50 pp |
| Hit@10 | 33.75% | 47.50% | +13.75 pp |
| Recall@10 | 27.29% | 43.33% | +16.04 pp |
| MRR@10 | 0.2648 | 0.3058 | +0.0410 |
| 平均延迟 | 100.18 ms | 99.68 ms | -0.50 ms |

真实 Pipeline 与缓存模拟结果一致，证明提升来自去重策略和检索/RRF 参数，而不是实验脚本与生产链路不一致。候选数量增加没有造成可见的平均延迟回归；Dense 使用精确向量搜索，当前 13,421 passages 的规模仍较小。

### 后续高成本实验

- embedding 模型替换可能需要下载模型并重建 FAISS 索引。
- passage 粒度变化需要重建 corpus、BM25 和 Dense 索引。
- 在前置低成本实验没有完成前，不同时改变模型和语料，避免无法归因。

## 实验 4：Embedding 模型对照

- 状态：GTE 尝试被运行时兼容性阻塞，转用 E5
- 当前模型：`BAAI/bge-base-en-v1.5`，768 维，query 使用官方 retrieval instruction，passage 不加 instruction。
- 第一对照：`alibaba-nlp-community/gte-base-en-v1.5`，约 137M 参数、768 维，按官方 Sentence-Transformers 用法不增加 BGE instruction。
- 选择原因：规模和向量维度相近，减少“仅因为模型大很多”的混杂；GTE 支持更长上下文，适合检验 history/长 section 是否是 BGE 的弱点。
- 固定变量：同一 passages、同一 80 样本、history query、BM25、domain filter、Dense Top 100、BM25 Top 50、RRF k=30、无 reranker、无额外去重。
- 模型来源与 revision 会固定在 `scripts/download_models.py`，权重 SHA-256 会校验，避免未来模型仓库更新造成不可复现。

GTE 尝试记录：模型文件已按 revision `532bc29f1fcb9f3d446568efdaa2c66af9255cf9` 下载并通过权重 SHA-256 校验，但项目锁定的 `transformers==4.48.3` 无法识别 checkpoint 的 `model_type="gte"`，在构建索引前失败。没有升级全局 Transformers，因为升级会同时改变 BGE 加载与推理行为，破坏单变量对照，并可能影响项目其他依赖。

替代方案：改用 `intfloat/e5-base-v2`。其官方模型卡明确要求非对称检索时 query 加 `query: `、passage 加 `passage: `，最大长度 512，与当前 BGE-base 的长度和模型规模更接近。为支持公平对比，Dense 索引构建需要增加独立的 passage prefix 配置，不能只修改 query prefix。

## 实验 5：BM25 文本处理与字段权重

- 状态：第一轮已完成
- 固定变量：BGE Dense Top 100、BM25 Top 50、RRF k=30、history query、无 reranker、无额外去重。
- 对比：baseline、移除角色标签、停用词过滤、bigram、标题重复加权及组合。

关键结果：

| BM25 变体 | BM25 Recall@200 | Hybrid Hit@10 | Hybrid Recall@10 | Hybrid MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 65.63% | 47.50% | 43.33% | 0.3058 |
| query 移除 `user/agent` | 65.63% | 48.75% | 44.58% | 0.3090 |
| 停用词过滤 | 78.13% | 45.00% | 40.83% | 0.3013 |
| unigram + bigram | 70.00% | 48.75% | 42.71% | 0.3062 |
| 标题额外重复 1 次 | 66.25% | 47.50% | 42.71% | 0.3090 |
| 标题额外重复 2 次 | 66.25% | 45.00% | 41.25% | 0.3259 |

发现：单路深层 Recall 提升不保证最终 RRF Top-10 提升。停用词过滤把 BM25 Recall@200 提高 12.5 个百分点，但改变了前部排序和与 Dense 的交集，最终 Hybrid Recall@10 反而下降 2.5 个百分点。bigram 也出现类似现象。

采取的改动：只在 BM25 查询侧移除人工角色标签 `user`、`agent`。该改动不改变语料 token，因此无需重建现有 BM25 pickle。停用词、bigram 和标题加权暂不进入生产默认配置。

## 实验 6：两路 Top-200 共同失败样本审计

- 状态：已完成第一轮人工检查
- 范围：80 条均衡样本中，旧 BGE 与 BM25 在各自 Top 200 都没有命中任何 Gold 的 15 条。

发现的失败类型：

1. **明显脏数据/无语义 utterance**：至少 4 条。真实 query 由 `a`、`hg`、`gh` 等占位字符组成，模型没有足够信息恢复 Gold。示例 `40dff...:6`、`:7`、`c579...:1`、`:4`。这类样本不应被普通 query rewrite 或换模型“修复”，应单独报告 clean/noisy subset。
2. **对话突然切换到新的导航主题**：多条 VA 样本从上一主题跳到 `Pension`、`Health care`、`Family member benefits`、`Burials and memorials` 等首页导航 section。history 中旧回答很长，新主题表达很短，支持“history 有帮助但也会污染主题切换”的猜想。
3. **间接表达或强指代**：例如 `What kind of care will my children get?` 对应 survivor benefits；`receive compensation because of my ex-wife` 对应 former spouse benefits。需要 standalone rewrite 或领域同义表达能力。
4. **Gold/问句语义存在可疑错位**：例如询问 `survivor benefits` 的样本 Gold 映射到 `Burials and memorials` section。该样本可能反映标注上下文、section 映射粒度或数据本身的问题，不能简单解释为模型失败。
5. **短标题/导航型 Gold**：不少 Gold 同时包含正文 section 和只有标题文字的 `t_*` section。标题型 passage 信息量很少，可能导致多 Gold coverage 被压低。

阶段结论：共同失败的 15 条并非单一模型问题。至少约四分之一是明显脏 query，剩余样本集中在主题切换、指代/间接表达、导航型 section 和潜在 Gold 错位。后续应同时报告原始 validation 指标与过滤明显无语义 query 后的 clean-subset 指标，但不能静默删除失败样本。

由失败审计新增两个无需生成模型、可完全复现的 query 对照：

- `user_history`：保留同一 4-turn 窗口中的 user utterance，移除 agent 长回答。
- `current_first_user_history`：当前 user 问题置前，再附加窗口内更早的 user utterance，减少长文本截断或开头主题对 Dense 表示的影响。

这两种方法属于上下文裁剪/重排，不是真正的 standalone query rewrite。实验报告必须保持这一责任边界。

### 查询上下文对照与留出验证

第一批 80 条均衡样本、固定正式候选参数（Dense 100 / BM25 50 / RRF 30）结果：

| Query 构造 | Hit@10 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: |
| history | 48.75% | 44.58% | 0.3090 |
| current | 45.00% | 42.08% | 0.2418 |
| user_history | 50.00% | 46.46% | 0.2769 |
| current_first_user_history | 50.00% | 46.46% | 0.3093 |

若直接在这 80 条上同时挑 query 与参数，最佳组合为 `current_first_user_history + Dense 30 + BM25 200 + RRF 60`，Hit@10 58.75%、Recall@10 55.21%。但这是在同一批样本上从大量组合中选择的结果，存在明显过拟合风险。

因此新增 `--offset`，用按相同规则抽取的下一批 80 条做留出验证。固定正式参数时：

| Query 构造 | 留出 Hit@10 | 留出 Recall@10 | 留出 MRR@10 |
| --- | ---: | ---: | ---: |
| history | 67.50% | 62.50% | 0.3630 |
| current | 51.25% | 48.75% | 0.3055 |
| user_history | 62.50% | 59.38% | 0.3504 |
| current_first_user_history | 58.75% | 55.62% | 0.3568 |

第一批最优组合在留出集只有 Hit@10 58.75%、Recall@10 56.25%，而现有 `history + 100/50/30` 为 67.50%/62.50%。结论是 query 裁剪效果依赖样本分布，不足以替换默认 history；“关闭子查询/去掉 agent history 一定更好”的猜想被否定。生产默认保持 history。后续若做真正 standalone rewrite，应使用独立开发集选参数，再在未参与选择的测试集报告结果。

## 实验 7：Passage 粒度统计与切分决策

- 状态：完成统计，暂不执行更细 chunk 重建

13,421 个 section-level passages 的正文词数：

| 统计量 | 词数 |
| --- | ---: |
| 中位数 | 19 |
| P75 | 41 |
| P90 | 69 |
| P95 | 91 |
| P99 | 160 |
| 最大值 | 586 |

只有 40 个 passage 超过 256 词、3 个超过 512 词，因此“section 普遍过长、Dense 大量截断”不是当前主要瓶颈。贸然把所有 section 切得更细会增加语料规模、改变 Gold 映射，并可能让本来很短的证据进一步碎片化。

更显著的问题是短 passage：2,150 个 passage 少于 5 词。validation 全集的 6,254 次 Gold occurrence 中，1,827 次（29.21%）是 `t_*` 标题 section，798 次（12.76%）少于 5 词；80 条均衡样本中标题 Gold 占 30.36%。

新猜想：数据集把标题 span 和正文 span 标成不同 `id_sec`，当前 section corpus 因而同时生成标题 passage（如 `t_7`）和正文 passage（如 `7`）。标题 passage 本身不构成充分回答证据，却会增加多 Gold coverage 的分母。后续应做“保留原映射 vs 将 `t_N` 归并到同文档正文 section `N`”的评估定义消融，并由全组确认后才能改变正式指标。

## 优化过程中的工程异常：E5 索引构建

- 首次使用 `batch_size=32` 一次性编码全部 13,421 个 passage，模型完成约 420 个 batch 后，进程接近工具单次运行时限被终止，尚未来得及写出索引。
- 随后尝试 `batch_size=48/64`，进程在较早阶段退出，推测是一次性持有完整文本、模型和输出矩阵时触发了 macOS 内存压力；索引目录仍为空，因此没有把半成品误当成有效索引。
- 极小检查成功：E5 能正常加载，单条向量维度为 768，归一化后自内积约为 1。这排除了“模型下载损坏或当前 Transformers 完全不兼容 E5”。
- 改进方案：新增可恢复的 `experiments/build_dense_chunks.py`，以 500 条 passage、batch 16 分块编码；每块立即保存为 `.npy`，最后检查区间连续性、形状和有限值，再合并生成 FAISS 与 metadata。这样同时降低峰值内存，并避免中断后从头重算。
- 当前状态：27 个分块已覆盖全部 13,421 passages，并成功合并为 768 维 FAISS 索引；连续区间、数组形状、有限值与 corpus fingerprint 均通过检查。

### E5 对照结果

输出：`experiments/output/embedding_model_comparison.json`

| 配置 | Hit@10 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: |
| 当前 BGE + BM25（角色词优化前的可比基线） | 47.50% | 43.33% | 0.3058 |
| E5 + BM25 | 46.25% | 42.92% | 0.3106 |

E5 Dense 单路的 Hit/Recall 为：@10 41.25%/39.38%，@30 50.00%/47.50%，@50 58.75%/56.88%，@100 66.25%/63.13%，@200 76.25%/72.50%。与旧 BGE 的深层结果（@200 Hit 76.25%、Recall 71.25%）相比，E5 只在深层 Recall 上略有增加，没有转化为 Hybrid Top-10 提升。

结论：当前低 Recall 不能仅归因于 BGE 模型选择。E5 没有在固定语料、样本、RRF 和候选深度下胜出，因此不替换生产模型；BGE 继续作为默认模型。E5 索引保留在实验目录，便于复现，但不会被正式 Pipeline 自动加载。

## 本轮最终状态

### 进入默认配置的改动

1. 默认关闭 Cross-Encoder；仍可通过 `full` 模式显式启用。
2. 默认关闭额外精确文本、语义相似和每文档数量去重；RRF 仍按 passage ID 合并同一候选。
3. Dense Top-K 从 30 提高到 100，BM25 Top-K 从 30 提高到 50，RRF k 从 60 调为 30。
4. BM25 查询侧移除序列化产生的 `User` / `Agent` 角色词，不需要重建索引。
5. 默认 query 继续使用最近 4 turn history；查询裁剪保留为实验选项，不进入生产默认。
6. BGE 继续作为正式 embedding；E5/GTE 均未替换生产模型。

### 同一批 80 条样本的最终真实 Pipeline

输出：`artifacts/evaluation_rrf_optimized_roles_80.json`

| 指标 | 优化前 RRF | 最终 RRF | 变化 |
| --- | ---: | ---: | ---: |
| Hit@5 | 28.75% | 42.50% | +13.75 pp |
| Recall@5 | 22.92% | 36.67% | +13.75 pp |
| Hit@10 | 33.75% | 48.75% | +15.00 pp |
| Recall@10 | 27.29% | 44.58% | +17.29 pp |
| MRR@10 | 0.2648 | 0.3090 | +0.0442 |
| 平均延迟 | 100.18 ms | 101.54 ms | +1.36 ms |

最终拆分结果：User Gold Hit/Recall@10 均为 47.50%；Agent Gold Hit@10 为 42.86%，Recall@10 为 42.21%。两组均仍有改进空间，没有证据支持只优化其中一类。

### 没有进入默认配置的方案及原因

- Cross-Encoder：现有实验降低 Hit/Recall，且显著增加延迟；留待一阶段召回稳定后单独优化。
- 额外去重：每文档两段上限和语义相似过滤会误删 Gold。
- 停用词/bigram/标题加权：部分提高 BM25 Top-200，但没有稳定提高最终 Hybrid Top-10。
- E5：深层 Dense Recall 略高，Hybrid Top-10 反而略低。
- GTE：与锁定 Transformers 不兼容；为保持单变量实验没有升级全局依赖。
- current / user-only history：第一批样本看似改善，下一批留出样本没有稳定复现。
- 全量细粒度 chunk：绝大多数 section 很短，“普遍过长”猜想被统计否定。
- 标题 Gold 归并：可能改变评估定义，未经团队确认不得用来美化正式指标。

### 验证状态

- 单元与集成测试：54 passed。
- 唯一 warning 为 LangGraph 依赖的未来默认值变更，不影响本轮检索结果。
- 生产默认、实验模型和评估定义均保留明确开关/独立产物，避免混用。

## 2026-10-29：转入重排优化前的召回阶段展示

为判断 Cross-Encoder 是否具备重排空间，候选评估入口新增 `rrf` 模式，并在保留 @5/@10 的基础上新增展示 @50/@100/@200。这里的 RRF 候选是“两路各取到最大 cutoff 后融合”的重排候选池，不执行 Cross-Encoder 和额外去重；因此它衡量的是重排前的候选上限，不等同于线上当前 100/50 分支截断后的最终结果。

输出同时包括总体 Hit/Recall、User Gold Recall、Agent Gold Recall及逐 Gold 融合排名。后续重排实验必须先检查 Gold 是否已进入对应候选池：候选池中不存在的 Gold 不归责于重排模型。

80 条均衡样本实际结果：

| 重排前候选 | Hit@5 / Recall@5 | Hit@10 / Recall@10 | Hit@50 / Recall@50 | Hit@100 / Recall@100 | Hit@200 / Recall@200 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 36.25% / 30.42% | 45.00% / 39.79% | 58.75% / 56.25% | 63.75% / 60.62% | 67.50% / 65.62% |
| Dense | 32.50% / 28.33% | 42.50% / 37.50% | 58.75% / 55.00% | 66.25% / 61.88% | 76.25% / 71.25% |
| RRF 融合 | 41.25% / 36.04% | 47.50% / 43.33% | 61.25% / 58.13% | 67.50% / 63.12% | 75.00% / 72.50% |

RRF@50 已有 61.25% 的样本至少命中一个 Gold，RRF@200 为 75.00%。这给后续 Cross-Encoder 设定了候选池上限：重排 Top 50 最多只能在这 61.25% 的样本上命中；扩大到 Top 200 虽提高理论空间，但会显著增加 pair 数量和延迟，需要在效果与成本之间做消融。

## 标题 section 与正文 section 消融（400 条均衡 Validation）

### 问题与假设

数据中标题 section 会生成独立 passage，例如 `doc::t_8`，正文则可能是 `doc::9`。官方 Gold 如果指向标题，而系统检索到同标题下的正文，按原始 passage ID 比较仍会判为未命中。因此提出两个问题：

1. 原始 Recall 偏低有多少来自标题 ID 与正文 ID 的错位；
2. 删除独立标题 passage、把标题并入正文后，真实检索排序是否改善。

检查数据后发现，不能用“删除 `t_` 前缀”进行映射。标题编号和正文编号经常不一致，例如 `t_20` 后接正文 `21`，`t_8` 后接正文 `9`。最终采用保守映射：标题 section 后紧随正文 section，并且两者标准化标题完全一致时才映射。共发现 3,986 个独立标题 section，其中 3,455 个可可靠映射，531 个没有满足条件的正文，不进行强制映射。

### 保留的三种语料模式

- `independent`：默认行为，标题与正文保持独立，原始 `artifacts/` 不变；
- `merge`：不生成独立标题 passage，把可映射标题文字拼到首个对应正文的 `text`；
- `metadata`：不生成独立标题 passage，标题只保留在正文 `section_title` 与 `span_ids`，正文文本不重复标题。

替代索引写入 `artifacts_title_merge/` 和 `artifacts_title_metadata/`，不会覆盖原始索引。构建脚本为每个目录写入 `corpus_metadata.json`，明确记录模式、passage 数和标题映射统计。Dense 增量构建仅在 passage ID 与 `index_text` 完全一致时复用原向量；`merge` 复用 5,980 条、重新编码 3,455 条，`metadata` 复用全部 9,435 条正文向量。

### 双指标契约

评估增加显式参数 `--title-section-diagnostics`：

- 原有无前缀指标保持官方 passage ID 精确匹配，不修改含义；
- `merged_title_*` 是诊断口径，将可靠映射的标题 Gold 和检索结果都映射到正文 ID；
- 两套指标同时写入结果文件，归并诊断指标不得冒充官方指标。

### 原始索引上的问题量化

同一批四领域各 100 条、共 400 条样本：

| 指标 | 官方原始口径 | 标题归并诊断 | 差值 |
| --- | ---: | ---: | ---: |
| Hit@5 | 43.50% | 45.00% | +1.50 pp |
| Recall@5 | 36.46% | 43.25% | +6.79 pp |
| Hit@10 | 55.50% | 57.00% | +1.50 pp |
| Recall@10 | 49.42% | 55.13% | +5.71 pp |
| MRR | 0.3084 | 0.3244 | +0.0160 |

这证实了假设：一部分低 Recall 确实只是系统找到了同标题下正文，但没有命中标题 passage ID。Hit Rate 增幅较小而 Recall 增幅较大，说明很多样本原本已经命中了至少一个其他 Gold，标题错位主要影响“多个 Gold 覆盖多少”。

### 三种语料的正式消融

下面以 `merged_title_*` 作为三种语料间的公平诊断比较；替代语料不包含独立标题 ID，所以其官方精确 ID 指标天然偏低，仍保存在 JSON 中但不用于选择模式。

| 模式 | Diagnostic Hit@5 | Diagnostic Recall@5 | Diagnostic Hit@10 | Diagnostic Recall@10 | Diagnostic MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| independent（原始） | **45.00%** | **43.25%** | **57.00%** | **55.13%** | **0.3244** |
| merge | 44.00% | 42.00% | 55.25% | 52.75% | 0.3047 |
| metadata | 42.75% | 40.75% | 54.25% | 52.00% | 0.2949 |

### 结论

标题错位解释了原始 Recall@10 约 5.71 个百分点，但“删除标题 passage”不是更好的生产检索方案。原始 `independent` 在公平诊断口径的 Hit、Recall、MRR 上均优于两种归并语料；标题短 passage 可能对标题式或概念式查询提供了有效候选，删除后反而损失排序信号。`merge` 又比 `metadata` 略好，说明若必须移除标题 passage，把标题显式拼入正文比仅作为字段更有效，但仍不及原始结构。

因此默认语料和原始官方指标保持不变。归并映射只保留为诊断指标，用于区分“真正没找到相关 section”与“找到同标题正文但官方 ID 不一致”。

实验输出：

- `artifacts/evaluation_rrf_balanced_400_title_diagnostics.json`
- `artifacts_title_merge/evaluation_rrf_balanced_400.json`
- `artifacts_title_metadata/evaluation_rrf_balanced_400.json`

验证：完整测试 60 passed；另有 1 条 LangGraph 依赖弃用警告，与本次改动无关。
