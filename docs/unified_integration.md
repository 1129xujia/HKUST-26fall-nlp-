# 统一入口与接口联调

日期：2026-11-06。同步基准：远程 main 的 858b6d4；本地采用 fast-forward 同步，没有重复合并旧分支。

## 运行路径

```text
cs-rag ask / demo / run --config configs/integrated.yaml
  → QueryPlanner（成员三的现有规划器）
  → Part2RetrievalAdapter / HybridRetrievalPipeline
      Dense + BM25 并行 → RRF → CrossEncoderReranker
  → EvidenceResponseGenerator（成员三）
  → MainEvaluatorAdapter（匹配 v2 Feedback 的评价器）
  → OptimizationController（成员四）
      regenerate / rewrite_query / adjust_hybrid / rerank
  → 每轮重新评价 → 返回 / 澄清 / 准备人工转接
```

全流程共享一个 BudgetLedger，覆盖初始规划、检索、生成、评价和后续优化。模型/索引在构造 runner 时加载，运行延迟从请求开始计；不同请求共享只读模型，轨迹、预算和检索快照各自隔离。

`pipeline: integrated` 由原 build_orchestrator 工厂选择，使用同一个 CLI 和 RunResult。旧配置缺省使用 `pipeline: legacy`，保留原 D3 脚本演示。两者都通过同一结果保存与离线评价命令导出。

## 环境与准备

使用项目根目录的 Python 3.10 环境。当前工作区已经完成可编辑安装、依赖检查、官方数据预处理、模型下载与索引构建。

新环境可按以下步骤准备：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install setuptools==69.5.1 wheel==0.43.0
.\.venv\Scripts\python.exe -m pip install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[local]"
# requirements-local.lock.txt 记录本轮实际本地模型依赖；核心 mock 锁保持独立

# 只在本地 .env 设置 DEEPSEEK_API_KEY，不要把密钥写入 YAML
.\.venv\Scripts\cs-rag.exe download-data --config configs/integrated.yaml
.\.venv\Scripts\cs-rag.exe download-models --endpoint https://hf-mirror.com
.\.venv\Scripts\cs-rag.exe preprocess --config configs/integrated.yaml
.\.venv\Scripts\cs-rag.exe index --config configs/integrated.yaml
.\.venv\Scripts\cs-rag.exe doctor --config configs/integrated.yaml
```

现有模型下载命令同时提供离线 Qwen、MiniLM Embedding 和 MiniLM Reranker；统一 DeepSeek 模式只加载后两者。下载源、模型 revision 与资产校验仍由 registry 约束。

统一入口复用共享索引，Dense 使用 NumPy 精确检索；Part 2 的 BM25、RRF 和 Cross-Encoder 算法实际参与执行。这样保留当前 3798 个 passages、grounding 和已验证模型，并避免混装队友独立环境的 Torch/NumPy/Transformers 精确版本。Part 2 原生 FAISS/BGE 路径仍可在其独立环境运行，本轮没有把那条路径的实验成绩当成统一入口成绩。

## 三个配置

| 配置 | 数据/模型 | 用途 |
|---|---|---|
| integrated.yaml | 官方 train 派生 dev-mini；DeepSeek、真实 MiniLM、真实重排 | 完整质量循环 |
| integrated_single.yaml | 同上，保持相同规划/检索/生成设置 | 只关闭质量、优化与转接的单次对照 |
| integrated_mock.yaml | 16 条合成对话；mock LLM、HashEmbedding | 无密钥接口开发 |

`integrated_single` 是完整单次检索链路对照，不等同于原 Dense-only Single-RAG 配置。正式实验应分别标明这些设置。

```powershell
.\.venv\Scripts\cs-rag.exe ask --config configs/integrated.yaml --message "How can I replace a lost driver's license?"
.\.venv\Scripts\cs-rag.exe run --config configs/integrated.yaml --split mini --limit 4
.\.venv\Scripts\cs-rag.exe retrieve --config configs/integrated.yaml --query "license replacement" --full
.\.venv\Scripts\cs-rag.exe evaluate --config configs/integrated.yaml --run-dir results/<batch_id>
```

`ask --history history.json` 接受按时间排列的 `[{"role":"user","text":"..."},{"role":"agent","text":"..."}]`。`--no-save` 关闭结果落盘；默认保留完整运行记录。

## 契约与控制规则

- 数据入口只把 DialogueInput 交给在线流程；sample_id 在返回后关联，参考答案、Gold、领域标签和 grounding 留在离线评价侧。
- 统一路径的 Part 2 Evidence.passage_id 使用共享 Passage.citation_id。生成器将 E1/E2 映射回这些 ID，最终回复、候选、检索结果和 Gold 使用同一套键；原始 doc_id、span 和 offset 不被重写。
- Part 3 生成器同时用于初始生成和优化后的改答。改答包含旧回答、反馈和 revision，未另建一套与成员三脱节的生成逻辑。
- 输出必须符合状态枚举、引用集合和正文引用约束。格式错误最多额外生成一次纠正，使用相同证据与共享预算；原始响应保留在 model_calls，纠正调用标记 format_repair。再次失败明确返回组件错误。
- 评价器直接解析 v2 Feedback，包括 failure_types、evidence_sufficient、preferred_retriever。答案完整性和证据充分性分别判断。
- pass 不能覆盖负面诊断；必要用户信息缺失进入澄清。证据足够而回答遗漏时可改答；无效检索后有明确改答依据时使用剩余重试预算。明确缺证据时不会强行重新生成并放行。
- 领域默认不硬过滤，避免错误领域判断排除跨领域证据。统一入口使用独立改写查询，重排默认 identity；没有假装执行依赖 User: 序列格式的历史重排策略。
- DeepSeek 共用主项目客户端，不再默认寻找仓库外 baseline/query/transport.py。显式 legacy transport 注入仍保留兼容路径。截断或错误响应也记录已报告 Token；取消/网络故障无完整用量时标记 unavailable。
- 每次请求最多执行配置中的两次优化。受 Token 或时间预算约束停止，转接包中 delivered_to_human=false，不代表已对接真实人工平台。

## 实际联调记录

没有执行测试套件。以下是 CLI、接口反例与真实服务开发运行检查。

| 检查 | 实际结果 |
|---|---|
| 安装 | 根环境可编辑安装成功，pip check 通过；无新增推理依赖 |
| 原 D3 场景 | 12 个脚本场景全部受控完成 |
| 成员四离线 demo | 四个模拟优化分支均完成 |
| 统一 mock mini | 16 条无组件错误：10 answered、2 clarification_needed、4 escalated；不代表正确率 |
| 矛盾反馈 | 缺少必要用户信息的 pass 被转为澄清 |
| 评价契约 | 三个 v2 诊断字段可解析 |
| 回答遗漏 | 建议 regenerate 时进入改答；相同上下文且证据足够时也可在剩余预算内改答 |
| 失败用量 | 模拟截断响应的 612 Token 保留并计入账本 |
| 预算 | 极小 Token/时间限制下受控停止 |
| 真实四领域 | 3 answered、1 escalated，无组件错误；VA 样例发生混合检索调整和改答，达到重试上限后转接 |
| 真实单次模式 | 1 条 answered，0 次评价、0 次优化 |

最终四领域批次：`results/batch_06108cdda3a54777b9bd3483f1a14875`。

参考答案 F1=0.1307723，EM=0，Evidence Recall@5=0.25；平均请求约 6.36 秒、6844.5 个服务端报告 Token。样本很少，不能作为正式质量或性能结论。无外部事实标注，幻觉率、grounding accuracy、对话一致性与任务完成率保持 null。

较早真实 DMV 运行 `0af2c03b7679430ab0162bf8610b0e62` 实际执行了 regenerate，1 次检索、2 次评价后通过。初次批次还有生成状态格式错误，后续已明确状态枚举并增加有界格式纠正；历史结果未篡改。

可提交的摘要：[integration_summary.json](../reports/integration_summary.json)。完整本地结果在 results，包含逐次检索、所有候选、原始模型响应、Token、v2 诊断与动作、实际配置、索引清单及代码/Prompt 校验值。

## 下一阶段

当前完成的是 MultiDoc2Dial 上的统一入口与接口联调。正式基线/消融、扩大样本、评价阈值校准、独立事实标注和 HotpotQA 原生适配尚未完成。不能将在线评价器判为 pass 等同于离线答案正确。
