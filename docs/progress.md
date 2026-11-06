# 开发进度与交接

## 本轮范围

2026-11-06：本地 main 已快进同步到 858b6d4，并完成 Part 2、Part 3 与 optimization_v2 的统一入口和接口联调。
新增 integrated / integrated_single / integrated_mock 配置与 cs-rag ask；统一模型客户端、引用 ID、v2 诊断和全流程预算。
真实官方数据 mini 上 4 条受控结束（3 answered、1 escalated），单次模式 1 条完成；原 D3 的 12 个场景仍可运行。测试套件继续暂缓。
详见 [统一入口说明](unified_integration.md) 和 [实际运行摘要](../reports/integration_summary.json)。以下 D1–D3 条目保留为历史交付记录。

## D1 交付

- [x] 固定数据契约、Protocol 接口、最终状态和评估决策。
- [x] 确定当前设备与 Python 兼容版本，采用项目独立 .venv。
- [x] 确定离线 mock LLM 与 hash Embedding，不需要 API 权限。
- [x] 固定配置字段、随机种子、环境覆盖与相对路径规则。
- [x] 编写 12 个虚构 passages 和 16 条共享对话，含回答、澄清和转接场景。
- [x] 约定原始 grounding、证据组、离线标注与在线输入隔离。
- [x] 固定实验配置与指标口径，明确官方数据许可核验在正式接入前完成。
- [x] 按成员编号指定共享契约、模型、数据和编排的默认维护职责。

## D2 交付

- [x] 创建 src 包结构、pyproject.toml、配置、Prompt 与数据目录。
- [x] 实现配置加载、字段校验、JSON 序列化和 JSON Schema 导出入口。
- [x] 实现共享 JSONL 读取、mini 数据导出和官方 JSON 容器读取入口。
- [x] 实现 hash 模型适配器、临时向量索引、BM25 和并行检索。
- [x] 实现保留来源的去重、交替合并与 RRF，为检索模块提供可运行起点。
- [x] 启动状态机、基础预算检查、请求级用量与运行日志。
- [x] 提供单次 mock 规划、生成与 demo，便于后续模块替换。
- [x] 提供 doctor、config、schema、samples、preprocess、index、retrieve、demo、run 命令。

## D3 交付

- [x] 实现 EvaluatorAgent，使用可见对话、当前证据和候选回答输出结构化 mock 反馈。
- [x] 实现 DecisionPolicy，按四项阈值及回答/引用状态判断是否允许通过。
- [x] 实现 OptimizerAgent 与动作执行器，支持改写、子查询、检索参数调整、重新生成、澄清、转接。
- [x] 将规划、检索、生成、评价、优化连接到同一个有界编排器。
- [x] 分开记录评估调用、实际检索和已执行重试次数；最多两次优化。
- [x] 接入无新证据、无有效动作、重试上限、Token 与延迟预算停止。
- [x] 实现澄清回复与包含对话、查询、证据、缺失信息、反馈和已尝试动作的转接包。
- [x] 保存逐轮检索、候选、评分、动作及停止原因，并保留被拦截的草稿。
- [x] 提供 configs/d3.yaml，保留 default / baseline 单次模式。
- [x] 提供 12 个显式可控场景，脚本与离线标准答案隔离。
- [x] 更新 JSON Schema、README、架构和数据契约。

## D2 实际运行记录（2026-11-04）

以下为安装与命令行运行检查，没有创建或执行测试套件。

| 项目 | 实际结果 |
|---|---|
| 安装 | 项目在 .venv 中可编辑安装成功，命令入口可用；pip check 无依赖冲突 |
| 环境读取 | doctor 读取到 Python 3.10.14、12 个 passages、16 条对话 |
| 配置 | 默认配置、baseline 覆盖与 E7 profile 能加载并输出 |
| 数据读取 | preprocess 成功导出 12 个 passages 与 16 条 mini 样本 |
| 临时索引 | 保存 256 维 hash 向量、语料和校验清单，运行时可重新加载 |
| 检索 | 驾照补办查询返回 demo_dmv:replacement，并保留 Dense/BM25 来源 |
| 单次演示 | mini_005 完成 planning→retrieving→generating→completed，保存完整本地产物 |
| 批量运行 | 全部 16 条样例能通过单次 mock 流程；这不表示 16 条回答正确 |
| 基线配置 | baseline.yaml 下成功执行 2 条 mini 样例，仍明确标记为 hash/mock |
| 契约导出 | docs/contracts.schema.json 已生成 |

D2 观察到 mini_001 的省略问句可能返回错误领域证据，单次模式不会拦截澄清或转接请求。
这些历史记录不计作正式实验成功；D3 在其基础上加入了拦截和停止路径。

## D3 实际运行记录（2026-11-04）

- 12 个脚本场景全部完成命令行演示，包括通过、两类重试、澄清、转接、上限和预算。
- retry-retrieve：2 次评估、2 次检索、1 次重试。
- retry-regenerate：2 次评估、1 次检索、1 次重试；不完整草稿替换为证据摘录。
- retry-limit：3 次评估、1 次检索、2 次重试，随后输出转接包。
- no-new-evidence：第 2 次检索未获得新的最终证据时停止，不继续生成和评估。
- token-budget：第一条完成调用前的估算 Token 预留被拒绝，停止原因为 token_budget。
- latency-budget：延迟 mock Evaluator 在请求截止时被取消，停止原因为 latency_budget。
- 16 条普通 mini 样例使用 D3 配置能够运行，无场景脚本或离线标注输入在线组件。
- mini_013/014 输出澄清，mini_015/016 准备转接；mini_001 仍未修复检索问题并以 no_effective_action 停止。
- mini_004 的多需求回答仍不完整，达到重试上限；此结果保留为待解决限制。
- baseline 单次模式另行运行 2 条样例，仍为 0 次评估、1 次检索。

详细场景与命令见 d3_walkthrough.md；保存的场景运行摘要见 ../reports/d3_demo_results.json。
这些都是离线开发运行记录，没有创建或执行测试套件，也不构成正式准确率或幻觉率评价。

## D3 结束时未完成（历史记录，当前状态见上文）

- 正式 LLM / Embedding / Cross-Encoder 选型、权重或 API 接入。
- 官方 MultiDoc2Dial 下载、许可核验、正式轮次构造和 span 对齐。
- 真实意图识别、查询改写、子问题分解和答案生成。
- 真实语义质量评估与基于真实模型的优化效果验证；D3 目前仅实现 mock 规则与控制流程。
- 正式 Single-RAG、E0–E7 实验执行和评价指标代码。
- 测试套件及测试相关工作，按用户要求暂缓。

## D3 时规划的下一步 D4（历史记录）

1. 下载并核验官方 MultiDoc2Dial 版本、许可和可用划分。
2. 构造无未来轮次泄漏的样本，完成 grounding span 到 passage/证据组的映射。
3. 明确真实 LLM、Embedding 与服务预算，接入适配器；资源尚未指定时保留离线入口。
4. 以首批真实语料建立索引并运行真实规划、检索和生成样例。
