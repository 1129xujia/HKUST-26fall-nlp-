# 基于评价反馈的分流优化器 v2

该模块可独立调用，**具体接口未与最新generation evaluation对齐，未根据根据评价器最新结果测试优化效果**。

实现范围：结构化反馈、确定性分流、四类优化动作、请求级加权融合、候选缓存重排、生成修订指令、重新评价、共享预算、停止条件和优化轨迹。真实模型质量与正式数据集效果需另外验证；离线 demo 的脚本评分只用于验证控制流程。

当前生成、查询改写和评价共用主项目 DeepSeek 客户端，从本地 `.env` / 环境变量读取密钥，不默认依赖仓库外文件。三处组件支持显式注入 LLM。统一 CLI、配置与运行记录见 [统一入口说明](../../docs/unified_integration.md)。

## 1. 调用入口与输入

```python
action = OptimizationRouter().route(request)  # 只选择一个动作，无外部调用
result = await controller.run(request)        # 执行动作、生成、重新评价并返回最终结果
```

`request` 类型为 `OptimizationInput`。JSON 示例见 `input.example.json`；机器可读结构见 `contracts.schema.json`。本入口从“上游已生成并评价第一版回答”开始。

Schema 根引用输入黑口，动作和输出接口位于同一文件的 `$defs`，内部引用可独立解析。模型变更后可执行 `python -B -m optimization_v2.export_contracts` 重新导出。

| 字段 | 类型 | 约定 |
|---|---|---|
| `query` | main `QueryPlan` | 当前问题、可见历史、改写查询、领域、子问题；禁止传入 gold 或未来对话 |
| `response` | main `GeneratedResponse` | 上一版候选回答；引用采用当前 `Evidence.passage_id` |
| `evidence` | Part 2 `list[Evidence]` | 当前有序生成上下文，保留原 ID、正文、span、来源与分数 |
| `feedback` | `Feedback` | 兼容 main 的全部 `EvaluationFeedback` 字段，新增诊断字段可选 |
| `options` | `ExecutionOptions` | 当前请求的召回、融合、重排及最终证据数量设置 |
| `snapshot` | `RetrievalSnapshot \| None` | 可选的可信运行时缓存；缺省时首次检索优化需要重新召回 |
| `capabilities` | `Capabilities` | 调用方允许的能力；运行时进一步与实际组件能力求交集 |
| `limits` | `Limits` | 重试、Token、耗时、评分阈值及转接开关 |
| `budget` | `BudgetState` | **必须累计上游已用 Token 和耗时**，才能约束整个请求 |
| `retry_count` | `int` | 已执行的重试次数，继续累计，不重新归零 |
| `action_history` | `list[OptimizationAction]` | 历史 v2 动作，用于诊断与追踪 |
| `trace_id` | `str \| None` | 传给 Part 2 请求的追踪标识 |

### 反馈接口

沿用：`relevance / sufficiency / groundedness / dialogue_consistency` 四项 `[0,1]` 评分，以及 `decision / reason / missing_information / suggested_action / suggested_queries / clarification_question`。

可选扩展：

- `failure_types`：`answer_defect / query_misunderstanding / retrieval_coverage / retrieval_balance / evidence_ranking / missing_user_information`。
- `evidence_sufficient: bool | None`：明确区分证据不足与回答遗漏；`None` 表示评价器未提供判断。
- `preferred_retriever: dense | bm25 | None`：有诊断依据时建议偏重的检索分支。

`needs_optimization` 是只读派生属性 `decision == retry`，不接受第二个可写布尔字段。低分、无有效引用、明确证据不足或有未解决诊断的 `pass` 会先被本地 Policy 改为 `retry`，再分流。Policy 不声称能凭词面分数判定事实真伪。

## 2. 明确的分流规则

依次执行，命中后不再选择其他主要分支：

1. `decision=clarify/escalate` 保留上游终止决策；合格 `pass` 返回当前回答。
2. `missing_user_information` 输出澄清问题；达到重试上限则停止。
3. 有结构化诊断时，优先级为：查询理解偏差 → 召回覆盖/分支失衡 → 证据排序 → 回答缺陷。
4. 无结构化诊断时，映射 `suggested_action`；`add_sub_queries` 归入查询优化，`adjust_retrieval` 归入混合检索优化。
5. 两者均无时：无证据或证据充分性不足优先检索，否则重新生成。
6. 请求重新生成但明确证据不足时，改走检索分支；缺少相应能力时停止并报告 `unsupported_branch:*`，不假装执行。

仅凭最终证据的 `rrf_score/rerank_score` 无法可靠判断遗漏发生在召回还是重排阶段。结构化诊断应来自评价器或结合候选池的诊断服务；缺少诊断时采用上述保守规则。

## 3. 四个分支对其他模块的接口

所有模型组件均为异步接口，且接收**同一个** `BudgetLedger`。

| 分支 | 调用接口 | 输入 | 输出与后续流程 |
|---|---|---|---|
| 回答优化 | `generator.generate(GenerationRequest, budget)` | 原始问题、查询计划、证据、上一版回答、反馈、`Revision` | main `GeneratedResponse`，随后评价；不重新检索 |
| 查询优化 | `rewriter.rewrite(RewriteRequest, budget)` | 上述上下文、检索设置和动作历史 | `QueryUpdate(rewritten_query, sub_queries)`；重新检索→生成→评价 |
| 混合检索 | `retrieval.retrieve(query, options, snapshot, trace_id)` | 查询、请求级权重/配额、可选缓存 | `RetrievalSnapshot`；使用新证据生成并评价 |
| 重排优化 | 同一个检索适配器 | 排序策略、候选数、子查询开关等，以及缓存 | 复用召回池或重新召回，再排序→生成→评价 |
| 重新评价 | `evaluator.evaluate(EvaluationRequest, budget)` | 查询、当前有序证据、新候选回答 | `Feedback`，再次执行 Policy 与分流 |

### 回答生成

`LLMRevisionGenerator` 使用 `LLMClient.complete(task="generate")`，不传入客户端时自动构造 `DeepSeekLLM`。Prompt 实际包含 `previous_response / feedback / revision`，避免只用相同输入重复调用旧生成器。`Revision` 包含修订要求、缺失信息和评价原因。生成器输出必须有合法引用；无效引用停止为 `error`，不会发布该草稿。

### 查询改写

`LLMQueryRewriter` 优先执行评价器的 `suggested_queries`；否则调用现有 LLM `task="plan"`，提供诊断与上一轮上下文。只接受改写和子查询两个输出字段，不能覆盖原始问题、历史、意图和领域。子查询规范化、去重并限制为 4 条。当前不会自动放宽领域过滤；调用方应仅对可信业务路由启用过滤。

### 混合检索

`Part2RetrievalAdapter` 保留 Part 2 的 `RetrievalRequest` 契约，扩展设置独立放在 `ExecutionOptions` 中，不修改其冻结配置或源码。

- `dense_top_k / bm25_top_k`：分支召回配额，范围 `0–500`。
- `dense_weight / bm25_weight`：融合权重，范围 `0–10`，至少一条分支同时具有正配额、正权重。
- 加权 RRF：对每个查询分支贡献 `weight[retriever] / (rrf_k + rank)`，不直接混合不同量纲的原始分数。
- 路由器默认扩大可用分支配额和候选范围；有 `preferred_retriever` 时采用该分支 `2`、另一分支 `1` 的有界权重。
- `candidate_top_k` 上限 `500`；`final_top_k` 范围 `1–20`。参数均按请求保存。

### 证据重排

支持 `identity / current_first_user_history`、是否使用子查询、短确认句保护及候选范围。使用 Part 2 已加载的 Cross-Encoder 构造轻量请求级包装器，重用模型并加锁，不重复下载或修改共享模型设置。自定义排序器可按策略名注册，接口仍为 `rerank(main_query, sub_queries, candidates)`；该自定义对象自行实现对应策略。

未加载真实重排器且未注入自定义重排器时，能力为不可用。原 Pipeline 的重排加载回退说明保留在 snapshot.notes；优化器不会把 NoOp 当成真正重排。

## 4. 缓存、引用和旧模块适配

- 缓存保留截断前各分支的原始 `SearchHit` 排名，可重新融合并扩大排序候选范围，最终证据列表不是完整候选池。
- 缓存键包含主/子查询、实际领域过滤、召回配额、调用方提供的语料/索引版本和 Embedding 标识。改变这些条件重新召回；仅调权、重排或上下文数量可复用。
- `corpus_version` 必填，索引或模型资产更新后必须更新版本。snapshot 是可信的运行时对象，不应直接接受终端用户伪造的候选池。
- 主查询使用 `QueryPlan.rewritten_query`，子查询使用 `sub_queries`。领域默认只作为提示；显式 `domain_filter=True` 后才过滤，`unknown` 映射成 `None`。
- 输出保留原 `Evidence.passage_id`。向 main 评价器传递时，使用 URL-safe Base64 的可逆映射避开其 ID 冒号限制，引用同步转换；不通过拆分冒号猜测 ID。
- `MainEvaluatorAdapter` 使用专用 evaluation_v2.jinja2，直接解析 Feedback；答案完整性与证据充分性分别判断，v2 诊断不经过旧 Schema。
- `from_main_state(state, feedback, ...)` 接受 main 已有 `RunState`，转换当前证据、查询、参数、预算和历史动作。不把 main 的 `interleave` 悄悄改成 RRF；遇到该设置明确报错。v1 改写/子查询动作统一映射为查询分支，检索调整映射为混合检索分支，重新生成补充历史修订原因；原始参数细节仍可在 main 轨迹中追溯。
- 同步检索用工作线程适配异步接口。超时可停止等待；Python 无法强制终止已进入原同步模型的线程，该线程可能稍后完成。它只操作深复制的本轮数据，不回写已结束请求；取消控制能力取决于底层服务。

## 5. 输出与停止条件

`OptimizationResult` 包含最终 `response`、查询、证据、参数、反馈历史、候选历史、动作、trace、重试次数、新召回次数、重新评价次数、预算用量、snapshot 和停止原因。初始上游评价不计入 `evaluations`；`retrieval_rounds` 只计真实召回，不计缓存重排。`feedback_history` 还可能包含空证据的本地诊断。

- 仅经过 Policy 的 `pass` 发布候选答案与引用。
- 澄清直接返回问题；重试/预算耗尽、无有效动作或无改善时输出信息不足或结构化转接包，`delivered_to_human=False`。
- 不把被拒绝的最后一版草稿当最终回答；草稿保留在 `candidate_history`。
- 比较有序证据的 ID、标题和正文；同一批 ID 的排序变化可继续生成。参数变化但上下文和查询均未变时，以 `no_context_change` 停止。
- 模型调用前保守预留 UTF-8 字节数加消息开销及输出上限，返回后按 provider 的 TokenUsage 扣账。预留是估算，不是精确分词；实际超额仍立即停止。`max_output_tokens` 应与底层模型实际输出上限一致。
- Token 用量缺失、提供方失败或取消后的用量不明时，记录为未知并停止，不当作零成本。自定义模型组件须通过 ledger.complete 或 ledger.record_usage 记账；普通计算/脚本组件可不产生模型 Token。

默认最多优化两次，默认评分阈值沿用 main；阈值尚未通过正式实验校准。

## 6. 对接示例

以下假定 Part 2 Pipeline 已在应用启动时加载，`state/feedback` 来自上游真实运行。工厂自动读取已有 DeepSeek 配置，并让三处模型组件共用一个客户端：

```python
from optimization_v2 import (
    Part2RetrievalAdapter, create_deepseek_controller, from_main_state,
)

retrieval = Part2RetrievalAdapter(pipeline, corpus_version="your-index-and-model-version")
controller = create_deepseek_controller(retrieval=retrieval)
request = from_main_state(state, feedback, capabilities=retrieval.capabilities)
result = await controller.run(request)
print(result.response.answer, result.response.citations)
```

如果直接从 Part 2 对接，构造 `OptimizationInput` 时使用 `options=retrieval.default_options()`、`capabilities=retrieval.capabilities`，并填入上游实际预算；首次检索时保存适配器返回的 snapshot，优化时传回以复用。

### DeepSeek 配置与实际连通检查

- 默认使用主项目 DeepSeekLLM，模型为 deepseek-flash，密钥来自本地 .env / DEEPSEEK_API_KEY；不寻找 baseline/query/transport.py。
- 显式 config_path 可加载本地 JSON 配置；显式 legacy transport、transport_path 或 OPTIMIZER_DEEPSEEK_TRANSPORT 才进入旧兼容路径。
- 构造客户端不发请求；complete 或 --run 才调用服务。错误/截断响应也保留服务端用量，未知用量标为 unavailable。
- 初始模型调用及优化分支共享同一预算。

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
# 只检查已有配置，不发送请求、不打印密钥
python -B -m optimization_v2.deepseek_check
# 使用虚构样例真实调用改写、生成、评价三部分
python -B -m optimization_v2.deepseek_check --run
```

## 7. 早期独立验证记录（历史）

在 `main` 根目录执行，依赖使用现有项目环境，不需要新安装包：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
python -B -m unittest discover -s src/optimization_v2/tests -v
python -B -m optimization_v2.demo
```

测试覆盖四分支、分流优先级、低分假通过、加权融合效果、缓存失效与复用、并发参数隔离、引用映射、生成修订传递、重新评价、重试上限、预算与异常。demo 使用虚构政策、脚本生成器和脚本评价器；不代表真实回答质量提升。

本次验证记录（2026-11-06）：

- Python 3.12.7、Pydantic 2.8.2、NumPy 1.26.4 的现有环境中，60 项离线测试通过（包含新增的 13 项 DeepSeek 接口检查），四分支脚本演示此前通过。
- 导出 Schema 的 33 个内部引用全部可解析，输入 JSON 示例通过模型校验。
- 新目录之外的 168 个文件在实施前后的聚合 SHA-256 一致；原有代码与配置未改变。
- 已使用已有 DeepSeek 配置完成真实改写→生成→重新评价检查：三次调用均成功，返回用量共 `2152` Token，虚构样例补全截止时间后为 `answered / evaluation_passed`。这是连通与格式检查，不构成正式数据集质量评价。
- 未运行正式数据集实验或原 CLI 集成；项目锁定的另一组依赖版本未另建环境复测。现有 main 模型在当前 Pydantic 版本下会发出 `model_` 字段命名警告，该警告不影响本次验证，也未通过修改旧代码处理。

## 8. 文件职责

| 文件 | 内容 |
|---|---|
| `schemas.py` | 输入、反馈、动作、请求级参数、缓存与结果契约 |
| `router.py` | Policy、确定性分流及参数调整策略 |
| `retrieval_adapter.py` | Part 2 请求级扩展、加权 RRF、缓存与独立重排 |
| `adapters.py` | main 状态/引用适配、生成修订、查询改写、旧评价器适配 |
| `interfaces.py` | 四分支及评价器的异步 Protocol |
| `controller.py` | 动作执行、重新评价、停止与最终输出 |
| `budget.py` | 所有分支共享的 Token/耗时账本 |
| `deepseek.py` | 复用已有 DeepSeek transport、配置与用量，提供共享客户端工厂 |
| `deepseek_check.py` | 默认不发请求的配置检查及显式真实连通验证 |
| `demo.py` | 四分支离线演示 |
| `export_contracts.py` | 导出自包含的 JSON Schema 契约 |
| `tests/` | 离线契约与行为测试 |

## 9. 当前提示词与调用关系

| 任务 | 当前调用路径 | 源码/模板 |
|---|---|---|
| 初始生成及改答 | LLMRevisionGenerator → EvidenceResponseGenerator；统一初始生成也用 EvidenceResponseGenerator | customer_service_rag/part3.py |
| 查询改写 | LLMQueryRewriter → 共享 LLMClient | optimization_v2/adapters.py |
| 评价 | MainEvaluatorAdapter → 共享 LLMClient → Feedback | customer_service_rag/prompts/evaluation_v2.jinja2 |
| HTTP 请求 | 主项目 DeepSeekLLM；本模块默认客户端为兼容包装 | customer_service_rag/models/deepseek_llm.py |

分流、配额、权重及重排设置由本地规则执行。只有显式 Prompt 被发送，payload 不作为第二个上传通道。

评价输出四维分数、decision、failure_types、evidence_sufficient、preferred_retriever 和修订建议。生成输出 answer/citations/status；临时 E1/E2 映射回本轮证据 ID。越界及不一致正文引用被拒绝；生成格式错误最多纠正一次，仍受同一预算限制。

完整接口和本轮实际结果见 [统一入口说明](../../docs/unified_integration.md)。
