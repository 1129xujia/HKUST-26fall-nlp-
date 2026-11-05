# D1 架构与技术决策

状态：D1–D3 离线契约 v1。mock 质量循环已实现；正式模型和官方数据接入仍属于后续工作。

## 运行环境与选型

- 当前工作机：Windows、Python 3.10.14、NVIDIA RTX 3050 6GB。
- 最低 Python 版本调整为 3.10；3.11+ 也可使用。项目独立安装在仓库的 .venv 中。
- 用户已选择暂不指定正式模型，先完成离线开发环境。
- LLM 固定为 mock-extractive-v1：规划时拼接上下文，生成时摘录证据，评价与优化使用显式词面规则。
- Embedding 固定为 hash-token-v1：256 维、SHA-256 token hashing、L2 归一化。
- HashEmbedding 是词面开发替身，不是语义向量模型；不能用于正式 Dense 实验结论。
- 检索使用 NumPy 精确余弦检索和 rank-bm25；融合支持 RRF 和交替合并。
- Pydantic 负责契约、PyYAML 负责配置、Typer 负责命令行、Jinja2 维护 Prompt。
- 直接依赖固定在 pyproject.toml；requirements.lock.txt 记录本次 Windows/Python 3.10 安装解析到的依赖版本。
- 运行时无需网络、API 密钥或 GPU。首次安装依赖需要包源；本轮没有下载正式模型权重。

## 分层与依赖

1. schemas.py 与 interfaces.py 定义共享契约。
2. data/ 负责读取语料，离线参考答案由 EvaluationExample 单独保存。
3. models/ 通过 LLMClient 与 EmbeddingModel 接口提供模型能力。
4. planning/、retrieval/、generation/、quality/、optimization/ 通过契约协作。
5. orchestration/ 控制状态和调用顺序；模型实例和用量按请求创建。
6. cli.py 负责参数解析，observability/ 负责保存配置与运行产物。

只有离线 evaluation/ 可以读取标准答案。CLI 从 EvaluationExample 中取出 dialogue 后再调用在线流程，不能把整个带标注样本传给智能体。

## D2 已运行的流程

DialogueInput → mock 查询规划 → Dense/BM25 查询分支并行执行 → RRF/交替合并
→ 首条证据摘录 → 引用 ID 校验 → RunResult 与本地日志。

- 相同文本的原始、改写和子查询会去重，避免同一查询重复贡献 RRF 分数。
- 分支结果保留 passage、检索器、查询 ID、原始分数和排名。
- 默认不启用重排序、在线质量检查、优化循环和自动转接。
- D3 配置可启用 mock 评估、优化与转接；重排序和初始子问题分解仍会报告尚未实现。
- 当前 demo 可能对澄清和转接样例给出不符合预期的摘录；D2 不把它解释为任务成功。

## D3 已运行的流程

规划 → 检索 → 生成 → EvaluatorAgent → DecisionPolicy。
pass 返回当前候选，clarify 输出问题，escalate 准备转接包，retry 进入 Optimizer。
动作由独立执行器校验和应用，下一轮选择重新检索或仅重新生成。

- default.yaml / baseline.yaml 维持单次模式，d3.yaml 开启质量闭环。
- LoopOrchestrator 通过组件工厂注入 Evaluator、Optimizer、Generator，每次请求创建新实例。
- 普通模式通过同一个 MockLLM 记录规划、生成、评估与优化用量。
- 场景模式明确注入 ScriptedEvaluator / ScriptedOptimizer / ScriptedGenerator。
- 场景脚本是可控服务返回值，不包含参考答案或数据集 gold；结果记录 mock_scenario。
- Policy 不允许低分或无有效回答的候选直接通过，即使脚本返回了 pass。
- 重新检索以最终供生成使用的 passage ID 集合判断新证据；仅重新生成不触发该条件。
- 不修改查询或参数的无效重复动作会以 no_effective_action 停止。
- 原始反馈与 Policy 后的反馈都保存在轨迹中，最终反馈另存 feedback.jsonl。
- 通过脚本返回 pass 只说明控制路径可运行，不能证明回答改善。

## 临时索引

索引目录包含 manifest.json、passages.jsonl 和 vectors.npy。

- manifest 保存格式版本、语料校验和、向量文件校验和、维度、模型标识和 passage 数量。
- 加载时校验文件、维度和模型；运行前再次比较配置语料与索引语料。
- BM25 从索引中保存的 passages 重建，不使用不透明的 pickle 文件。
- 修改语料或向量维度后，需要重新执行 cs-rag index。
- D2 索引只针对小语料；正式大规模索引和 Cross-Encoder 在后续阶段接入。

## 状态与预算

状态：created、planning、retrieving、generating、evaluating、optimizing、completed、failed。
合法转移由 orchestration/state.py 的 TRANSITIONS 固定，终态不能继续转移。

- D2 实际经过 planning、retrieving、generating、completed。
- iteration 表示已发起的候选答案评估调用次数，超时可能没有对应反馈；单次模式为 0。
- retrieval_round 表示实际检索次数，正常 D2 单次流程为 1。
- retry_count 只统计已执行的检索或生成重试，最多为 2；终止类动作不增加重试数。
- D3 在阶段间检查预算，通过 wait_for 控制请求截止，并在 mock 完成返回前检查估算 Token 预留。
- 预算、无新证据或重试耗尽时不返回被拦截的候选，按 escalation.enabled 输出转接或信息不足。
- 正常通过、澄清、转接及受控停止都进入 completed；组件异常进入 failed 并返回 error。
- 真实服务的 Token 估算和输出上限仍需另行接入与校准。
- Mock Token 数按字符数近似估算，usage.source=estimated；缺失用量不能记为零。
- 人工转接包始终标明 delivered_to_human=false，直到真实平台确认接收。

## 默认维护约定

维护人先按任务文档中的成员编号分配，姓名可在组员确定后补充。

| 共享内容 | 默认维护人 | 配合方 |
|---|---|---|
| schemas.py、interfaces.py | 成员四 | 全组共同评审字段变化 |
| config.py、cli.py、构建配置 | 成员四 | 全组提供模块配置 |
| 数据契约、样本、离线指标协议 | 成员一 | 成员二确认 grounding 与索引映射 |
| Embedding 与检索参数 | 成员二 | 成员一提供语料，成员三提供查询 |
| LLM 适配器与生成 Prompt | 成员三 | 成员四复用模型调用接口 |
| 在线质量检查、循环与日志 | 成员四 | 全组集成 |

## 后续接入顺序

1. D4：官方对话轮次构造、grounding 对齐、真实模型服务配置。
2. D5–D6：真实查询改写、分类、子问题分解、Reranker 和评价脚本。
3. D7–D9：正式索引、真实组件与已有循环集成，以及异常处理完善。

生产模型的提供方、权重、版本、预算和官方数据许可不在本轮虚构确认。
