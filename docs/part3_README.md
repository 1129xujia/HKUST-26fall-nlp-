# 成员三：对话理解与回答生成

## 职责与流程

本模块负责两个节点：

1. **Query Planner**：读取当前问题和可见对话历史，生成独立可检索的 `query`，以及可选的 `sub_queries`、`domain`。
2. **Response Generator**：读取 Part 2 已排序的 `Evidence`，按原顺序组织上下文，生成回答和可核对的引用。证据不足时返回 `insufficient_evidence`。

```text
当前问题 + 历史对话 → Query Planner → Part 2 Retrieval → Evidence → Response Generator → Evaluator
```

本模块只依赖其公开的 `RetrievalRequest`、`Evidence` 和 `make_retrieval_node`，不调用 Dense/BM25 或读取 FAISS 行号。候选回答的质量判断、重试、澄清和转接由 Evaluator/Optimizer 与编排器负责。

## 接口约定

- 上游输入：`DialogueInput(conversation_id, current_message, dialogue_history)`；历史记录按时间顺序排列。
- 规划输出：`QueryPlan`。交给 Part 2 时，将 `rewritten_query` 映射为 `RetrievalRequest.query`；`domain="unknown"` 映射为 `None`。
- 检索输出：Part 2 的 `list[Evidence]`，或 `make_retrieval_node` 返回的 `{"evidence": [dict, ...]}`。
- 生成输出：`GeneratedResponse(answer, citations, status)`；`citations` 使用 Part 2 的 `Evidence.passage_id`，只能来自本轮 Evidence。主项目 mock 的旧格式 `doc_id:passage_id` 在边界处规范化，不向下游输出。

主仓库现有 `QueryPlanner`、`ResponseGenerator` 和 Prompt 模板是成员三的基础实现。当前接口层可接收现有 `QueryPlanner`；因为主仓库 `ResponseGenerator` 只接受自己的 `Passage` 类型，而 Part 2 输出 `Evidence`，这里另写了 `EvidenceResponseGenerator`，直接按 Part 2 的公开字段生成。全组已选 DeepSeek，`models/deepseek_llm.py` 提供正式 API 适配器；离线 mock 仍用于无需 API Key 的接口回归。正式 Prompt/生成质量实验尚未运行。

## 计划与验收

1. 建立从 `QueryPlan` 到 `RetrievalRequest`、从 `Evidence` 到生成器输入的显式转换。
2. 提供一次完整的规划→检索→生成调用；在无真实索引时可用模拟检索节点调试。
3. 验证改写查询、领域映射、证据顺序、引用合法性和空证据处理。
4. 在统一样本上比较仅当前问题、完整历史、Query Rewriting 和子问题分解设置；正式实验结果在实际运行后补充。

## 当前实现与运行

`src/customer_service_rag/part3.py` 提供 `to_retrieval_request`、`evidence_from_node`、`EvidenceResponseGenerator` 和单轮 `run_once`。调用方在启动时构造 Part 2 pipeline，再将 `make_retrieval_node(pipeline)` 传入 `run_once`；模型选择沿用主仓库的 `LLMClient`。接口层不负责循环重试。

安装主项目依赖后，在项目根目录运行；完整预期结果和失败解释见 [`tests/README.md`](../tests/README.md)：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -p "test_part3_*.py" -v
python scripts/verify_part3_behavior.py
```

测试使用固定 Planner、Part 2 提供的真实 `make_retrieval_node`（注入固定 Pipeline）和模拟模型，不需要 API Key、真实索引或 GPU。当前已验证规划字段传递、Evidence 顺序、引用映射、空证据和越界引用拒绝。


### 调用示例

下面展示正式索引就绪后的单轮接法；模型可先用 mock，联调时再通过 `configs/part3_deepseek.yaml` 选择全组确定的 DeepSeek 适配器。Part 2 pipeline 在程序启动时创建一次，不要每轮重复加载。

```python
import asyncio

from customer_service_rag.config import PlanningConfig
from customer_service_rag.models.llm import MockLLM
from customer_service_rag.part3 import EvidenceResponseGenerator, run_once
from customer_service_rag.planning.planner import QueryPlanner
from customer_service_rag.schemas import DialogueInput
from retrieval_v2.retrieval import HybridRetrievalPipeline, RetrievalConfig
from retrieval_v2.retrieval.graph_node import make_retrieval_node

llm = MockLLM()
pipeline = HybridRetrievalPipeline.from_artifacts(
    artifact_dir="src/retrieval_v2/artifacts",
    config=RetrievalConfig.for_mode("bm25", artifact_dir="src/retrieval_v2/artifacts"),
    device="cpu",
)
dialogue = DialogueInput(
    conversation_id="demo-1",
    current_message="Can I apply again?",
    dialogue_history=[{"role": "user", "text": "I received student aid last year."}],
)
result = asyncio.run(run_once(
    dialogue,
    planner=QueryPlanner(llm, PlanningConfig()),
    retrieval_node=make_retrieval_node(pipeline),
    generator=EvidenceResponseGenerator(llm),
))
print(result.request, result.response)
```

此示例要求 `artifact_dir` 中已有 Part 2 生成的 passages 与 BM25 索引；当前 Git 仓库只保留 `.gitkeep`，所以没有索引时应先运行离线测试，而不是期待示例直接返回答案。`run_once` 只产生候选回答，不替代成员四的质量决策。

### 与成员四的待对接点

`Part3Result.evidence` 保留 Part 2 的 `Evidence` 类型，`response.citations` 使用其中的 `passage_id`。主项目原有 Evaluator 接口接收另一种 `Passage`，其引用键为 `doc_id:passage_id`；不能把这两种类型与引用规则直接混用。成员四接入循环前需与全组统一评价输入（或建立显式适配），并验证引用检查、重试和最终状态。此项不属于当前单轮交接的已通过测试。

## 交付状态

- [x] 成员三与 Part 2 的公开接口连接；单轮生成和引用 ID 校验；离线契约测试。
- [ ] 使用真实索引和正式 LLM 完成端到端运行，并记录可复现的模型、Prompt、样本与结果。
- [ ] 在 MultiDoc2Dial 上完成查询改写、历史上下文和子问题分解消融；保存逐样本输出与指标。
- [ ] 与 Evaluator/Optimizer 接入完整循环，确认不合格答案不会直接作为最终答复。


## 离线行为检查结果

`tests/fixtures/part3_behavior_cases.json` 使用全部 16 条已有 mini 样例和 2 条补充案例，人工标注查询必须包含的要素和预期领域。其中 12 条还检查回答要素与所需引用。`scripts/verify_part3_behavior.py` 实际调用当前 `QueryPlanner` 和 `EvidenceResponseGenerator`；生成阶段注入标准证据，因此不测检索效果。逐项结果保存在 `reports/part3_behavior_mock.json`。另外 10 项单元测试验证规划器、Part 2 节点边界、引用和 DeepSeek 适配器的离线行为。

当前 mock 运行 **70/78 项检查通过**。其余 8 项为：6 条有历史对话的查询仍带有 `Conversation context:` / `Current question:` 前缀，并非真正独立可读的改写；`mini_004` 的回答只用了首条证据，遗漏截止日期及第二个引用。检查时还发现 `mini_010`、`mini_015` 的 `ssa` 领域识别遗漏，已在 `planning/routing.py` 补充明确的付款与银行信息线索，复测通过。

对 18 条输出的逐例语义复核见 [`reports/part3_semantic_review.md`](../reports/part3_semantic_review.md)：15 条可判断查询中 9 条独立可读；注入正确证据的 12 条回答中 11 条完整且有依据。

这些结果说明当前 mock 能验证接口和部分规则，也不是正式模型在 MultiDoc2Dial 上的效果评分。正式模型接入后应在同一批案例上重跑，再进行官方数据实验。
