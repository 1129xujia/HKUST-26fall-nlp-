# 架构与接口

## 一条主流程

```text
DialogueInput
  -> QueryPlanner
  -> ProjectRetriever / customer_service_rag.retrieval.HybridRetrievalPipeline
       Dense + BM25 -> RRF -> CrossEncoder
       或 finetuned Dense 直接排名
  -> EvidenceResponseGenerator
  -> customer_service_rag.optimization.MainEvaluatorAdapter
  -> OptimizationController + OptimizationRouter
       改写查询 / 调整召回 / 重排 / 改答
  -> 回答、澄清或人工转接包
```

`src/` 只包含 `customer_service_rag` 一个源码包，队友的检索与评价优化实现分别位于其 `retrieval` 和 `optimization` 子包。唯一应用编排器是 `customer_service_rag.orchestration.orchestrator.RAGOrchestrator`，CLI、数据集运行和原生profile验收均复用它。队友模块的核心算法保留；旧legacy/D3链路、独立演示、重复生成器和微调runtime副本已移除。

## 模块职责

| 模块 | 职责 |
|---|---|
| `customer_service_rag.data` | 官方数据处理、样例、grounding和共享索引校验 |
| `customer_service_rag.planning` | 上下文理解、查询改写、领域识别和子问题 |
| `customer_service_rag.retrieval` | 队友实现的检索器、融合、重排、切分与原生索引 |
| `customer_service_rag.orchestration.retrieval` | 共享/原生资产适配，转换为同一 Evidence 与 RunResult |
| `customer_service_rag.generation` | 队友的证据限定生成与引用校验 |
| `customer_service_rag.optimization` | 队友的控制器、诊断路由、共享预算、检索快照和评价适配 |
| `customer_service_rag.models` | Embedding、LLM 和 mock 模型适配 |
| `customer_service_rag.evaluation` | 与在线输入隔离的参考答案、证据、引用和成本评价 |
| `customer_service_rag.observability` | 配置、代码/Prompt指纹和完整结果记录 |

## 两种资产后端

`retrieval.backend: shared` 复用现有 NumPy/MiniLM 或 hash 索引。SharedDenseRetriever只将共享向量暴露给队友的search协议，LocalPairPredictor只适配已校验的本地权重；BM25、RRF、CrossEncoder和优化召回缓存都使用队友实现。

`retrieval.backend: native` 加载队友的 FAISS/BGE profile。segmentation选择section/struct，chain选择rerank/finetuned。所有后端都交给同一个 Part2RetrievalAdapter 和 OptimizationController，模型及只读索引在启动时加载一次。微调首轮查询保持最近4个可见turn及角色标签，匹配训练表示。

共享引用为 `doc_id:passage_id`，原生引用保留 `doc::section/structure_N`。共享Passage只在明确的 `source=native_section/native_struct` 时接受原生ID，citation_id原样返回。原生Gold通过离线grounding span映射到对应原生语料，核对语料指纹后评价；不能拿字符窗口Gold直接计算section/struct召回。

## 输入、输出和控制

- 在线只接收当前问题和可见历史。参考答案、Gold领域和grounding仅在运行返回后参与评价。
- 模型面对E1/E2别名，生成器返回前映射回真实citation ID，并检查状态、引用集合和正文引用。格式错误最多纠正一次，仍使用原证据与共享预算。
- 一个BudgetLedger覆盖规划、初次生成/评价和所有优化调用。最多两次优化；重试、检索与评价分别计数。失败响应已报告的Token仍计入，用量不可得时保持unknown。
- 回答完整性与证据充分性分开诊断。需要用户信息时澄清；无法执行的操作或不足以可靠回答时停止，并按配置准备转接包。包中的delivered_to_human=false不代表已发送给真实客服。
- 每次请求的反馈、预算、快照与轨迹独立；共享模型只读并在必要位置加锁。

## 配置和运行记录

`configs/default.yaml` 是离线完整流程，deepseek是真实共享模型，native选择原生模型，experiments列出基线及循环消融。配置统一收敛到retrieval节点；不再使用pipeline或integration开关。历史已展开配置中的这两个旧字段只在读取时转换。

`RunResult`继续兼容历史JSON字段。新结果的pipeline_backend为unified，integration保存实际后端、模型/语料身份、原生诊断、动作和Token账本。metadata额外保存实际组件和Prompt的SHA256。历史实验结果和checkpoint来源记录保留原内容；迁移路径不会改写当时的配置。
