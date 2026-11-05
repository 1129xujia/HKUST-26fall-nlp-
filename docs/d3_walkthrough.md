# D3 评估—优化闭环

状态：已完成离线控制流程，未接入真实模型或官方数据。测试套件按用户要求暂缓。

## 使用方式

在项目根目录使用虚拟环境中的 cs-rag：

~~~powershell
# 如果索引尚未生成
.\.venv\Scripts\cs-rag.exe index

# 查看并运行显式可控场景
.\.venv\Scripts\cs-rag.exe scenarios
.\.venv\Scripts\cs-rag.exe demo --scenario all
.\.venv\Scripts\cs-rag.exe demo --scenario retry-regenerate

# 普通样例使用词面 mock 评估，不读取场景脚本
.\.venv\Scripts\cs-rag.exe demo --config configs/d3.yaml --sample-id mini_013
.\.venv\Scripts\cs-rag.exe run --config configs/d3.yaml --split mini

# 保留单次基线入口
.\.venv\Scripts\cs-rag.exe run --config configs/baseline.yaml --split mini --limit 2
~~~

demo 支持 --no-save 和 --full。--scenario 与 --sample-id 互斥。
场景模式默认使用 configs/d3.yaml，再合并 data/scenarios/d3.yaml 中可见的覆盖配置。
使用显式 --config 时仍要求 quality.enabled=true。

## 场景运行结果

以下是在 2026-11-04 执行命令观察到的控制流程。
脚本反馈是人工编写的服务返回值，不是在线组件读取标准答案，也不代表回答质量达标。

| 场景 | 最终状态 | 评估调用 | 检索调用 | 重试次数 | 停止原因 |
|---|---|---:|---:|---:|---|
| pass | answered | 1 | 1 | 0 | evaluation_passed |
| retry-retrieve | answered | 2 | 2 | 1 | evaluation_passed |
| retry-regenerate | answered | 2 | 1 | 1 | evaluation_passed |
| clarify | clarification_needed | 1 | 1 | 0 | clarification_required |
| escalate | escalated | 1 | 1 | 0 | evaluator_escalation |
| retry-limit | escalated | 3 | 1 | 2 | retry_limit |
| no-new-evidence | escalated | 1 | 2 | 1 | no_new_evidence |
| token-budget | escalated | 0 | 0 | 0 | token_budget |
| latency-budget | escalated | 1 | 1 | 0 | latency_budget |
| no-escalation | insufficient_evidence | 1 | 1 | 0 | retry_limit |
| rewrite-query | answered | 2 | 2 | 1 | evaluation_passed |
| add-sub-queries | answered | 2 | 2 | 1 | evaluation_passed |

retry-retrieve 实际把最终证据数从 1 扩展到 3，再由脚本模拟返回通过。
retry-regenerate 第一次返回不完整草稿，第二次摘录现有证据，没有再次检索。
rewrite-query 和 add-sub-queries 为了演示动作，显式关闭无新证据停止；正常配置仍开启。
latency-budget 在 mock Evaluator 中加入 0.2 秒异步延迟，配置请求截止为 0.05 秒；
负载较高时可能更早达到截止，但停止原因仍为 latency_budget。

本地详细产物位于 results/<run_id>/。场景运行摘要保存到 reports/d3_demo_results.json。

## 普通样例与脚本的区别

- 普通模式：Planner、Generator、Evaluator、Optimizer 通过 MockLLM 工作。
- Evaluator 只使用当前问题、可见历史、候选和提供的证据；不读取 reference_answers、gold_citations 或 expected_behavior。
- mock 规则能识别部分业务领域不一致、缺上下文、缺引用、证据主题遗漏和明确人工请求。
- Optimizer 根据反馈添加查询、扩大证据或请求重新生成，不发明政策内容。
- 场景模式：显式注入 ScriptedEvaluator、ScriptedOptimizer、ScriptedGenerator，结果包含 mock_scenario。
- 固定脚本返回本身不产生 LLM Token；usage 只记录实际调用 MockLLM 的估算值。
- 16 条普通样例可完成循环，但 lexical pass 仍可能漏掉语义错误，不计算正式准确率。
- 当前 mini_001 仍无法正确修复检索，mini_004 仍无法完整回答多需求；分别受控停止，不返回被拒绝的草稿。

## 停止与返回约定

- pass：返回当前通过策略的候选答案。
- clarify：返回具体澄清问题，不增加重试次数。
- escalate：根据配置准备转接包；关闭转接时返回信息不足。
- retry：最多两次有效检索或生成重试，之后最多共三次评估调用。
- no_new_evidence：第二次及后续检索没有新增的最终证据，停止，不继续生成。
- no_effective_action：查询或参数没有实际变化，停止。
- optimization_disabled：评价不通过但配置不允许优化，停止。
- token_budget / latency_budget / token_usage_unavailable：停止模型调用，不返回之前被拦截的答案。
- component_error:*：记录异常类型与阶段，返回 error，命令退出码为 1。

正常通过、澄清、转接和受控停止的退出码为 0；参数、配置或组件错误使用非零退出码。
受控停止进入 completed，组件异常进入 failed。状态机终态不能继续转移。

## 日志与转接

RunResult 分开保存最终回复、全部候选、每轮检索、反馈与实际动作。
traces.jsonl 包含阶段转移和阶段完成事件，并保存原始反馈与 Policy 后的反馈。
反馈写入 feedback.jsonl，动作写入 actions.jsonl，草稿写入 candidates.jsonl，
retrieval.jsonl 按实际检索轮次保存。脚本场景额外保存 scenario.json。

转接包包括原因、当前问题、历史、查询、已收集证据、缺失信息和已尝试动作。
delivered_to_human=false；当前没有连接真实客服平台。

## 后续工作

D4 接入官方数据的轮次和 grounding 映射，并在模型资源确定后替换离线适配器。
真实语义评分、质量指标、Cross-Encoder 和正式消融实验仍需后续实现。
