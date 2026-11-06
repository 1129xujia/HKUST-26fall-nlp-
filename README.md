# 在线客服多智能体 RAG 系统

HKUST 26fall NLP Course Project

基于并行检索融合与评估—优化循环的在线客服多智能体 RAG 系统。

> 当前阶段：D1–D3 离线工程与 mock 评估—优化闭环已完成。支持通过、重试、澄清、转接与受预算约束的停止，并提供 12 个可控场景。真实模型、官方数据与正式质量评价仍待接入；测试套件按要求暂缓。

## 快速开始（当前可运行）

以下命令在项目根目录执行。已在 Windows / Python 3.10.14 的独立虚拟环境完成安装；首次安装依赖需要包源，安装后运行无需网络、API 密钥或 GPU。

~~~powershell
# 首次使用时创建环境并安装；当前工作区已经完成
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c requirements.lock.txt -e .

# 查看配置、环境和共享样例
.\.venv\Scripts\cs-rag.exe doctor
.\.venv\Scripts\cs-rag.exe config
.\.venv\Scripts\cs-rag.exe samples

# 导出 mini 数据并建立临时索引
.\.venv\Scripts\cs-rag.exe preprocess
.\.venv\Scripts\cs-rag.exe index

# 查看检索结果与单次离线演示
.\.venv\Scripts\cs-rag.exe retrieve --query "Which form replaces a lost driving license?"
.\.venv\Scripts\cs-rag.exe demo --sample-id mini_005

# 运行全部 16 条共享样例；输出不代表回答正确率
.\.venv\Scripts\cs-rag.exe run --split mini

# D3：查看并演示全部控制路径（使用显式脚本模拟服务反馈）
.\.venv\Scripts\cs-rag.exe scenarios
.\.venv\Scripts\cs-rag.exe demo --scenario all

# D3：仅重新生成场景，评估两次但只检索一次
.\.venv\Scripts\cs-rag.exe demo --scenario retry-regenerate

# D3：在共享样例上使用普通 mock 规则，无脚本反馈
.\.venv\Scripts\cs-rag.exe run --config configs/d3.yaml --split mini
~~~

默认使用 `mock-extractive-v1` 与 `hash-token-v1`。回答是证据摘录，评估采用词面 mock 规则，HashEmbedding 是开发替身。场景里的脚本评分只用于演示流程，均不代表真实模型质量。合成政策内容全部虚构。运行文件保存到 `results/<run_id>/`，完整结果见 `result.json`；终端加 `--full` 可查看 demo 或 retrieve 的完整对象。

详细交接：[开发进度](docs/progress.md)、[D3 使用与验收记录](docs/d3_walkthrough.md)、[架构与技术决策](docs/architecture.md)、[数据契约](docs/data_contracts.md)、[实验协议](docs/experiment_protocol.md)、[成员三：对话理解与回答生成](docs/part3_README.md)。

## 1. 项目目标与范围

项目以 MultiDoc2Dial 为主要数据集，从零实现 Single-RAG 基线，并逐步加入对话理解、多路检索、证据融合、答案评估和循环优化能力。

核心目标：

1. 根据历史对话理解当前问题，完成查询改写、意图识别和可选的子问题分解。
2. 使用 Dense Retrieval 和 BM25 并行检索，通过 RRF 融合和 Reranker 筛选证据。
3. 生成直接、可执行、带文档或段落引用的客服回复。
4. 通过 Evaluator–Optimizer Loop 检查答案，在证据不足时调整查询、补充检索或重新生成。
5. 在无法可靠作答时请求用户澄清，或生成包含原因与上下文的人工转接结果。
6. 与 Single-RAG 比较，通过消融实验分析效果、延迟和 Token 成本。

第一阶段交付可通过命令行运行的研究系统、统一评价脚本、实验结果和报告。人工转接先实现结构化转接包；真实客服平台的连接、网页界面、HotpotQA 补充实验和第三种检索器作为后续扩展。

设计依据：工作区中的 `online_customer_service_rag_task_allocation.md` 与 `writing-block (3).md`。当研究范围调整时，应同步更新本 README 和实验协议。

## 2. 系统架构

采用 Python 模块化单进程架构。各智能体是具有独立职责和输入输出的逻辑模块，由 Orchestrator 统一调度。检索分支通过异步任务或线程池并行执行；具体实现根据模型服务和本地索引的调用方式选择。

```text
当前问题 + 历史对话
        │
        ▼
Context / Query Planner
  改写查询、识别意图与领域、可选子问题分解
        │
        ▼
Parallel Retrieval
  Dense × 查询分支  ─┐
  BM25  × 查询分支  ─┴→ 去重与来源聚合 → RRF → Reranker
                                               │
                                               ▼
                                       Response Generator
                                         回答 + 引用
                                               │
                                               ▼
                                        Evaluator + Policy
                              ┌─────────┬─────────┬─────────┐
                            pass      retry     clarify   escalate
                              │         │          │         │
                            返回回答   预算检查    请求澄清   人工转接包
                                        │
                                        ▼
                                     Optimizer
                               产生结构化优化动作
                                        │
                          ┌─────────────┴─────────────┐
                      调整查询或检索参数          使用已有证据修改回答
                          │                           │
                       重新检索                    重新生成
```

循环控制约定：

- 初始执行一次，最多再优化两次，即最多三次候选答案评估。
- `iteration` 记录候选答案评估次数，`retrieval_round` 记录实际检索次数；使用已有证据重新生成时，检索次数不增加。
- 评价通过、需要用户澄清、达到重试上限、达到预算，或重复检索未获得新证据时结束循环。
- 缺少用户身份、申请类型等必要输入时优先澄清；缺少政策依据时补充检索，仍无法支持回答则输出信息不足或人工转接结果。
- Optimizer 只提出动作；Orchestrator 校验动作、执行动作并维护状态。

## 3. 推荐技术方案

离线开发选型已固定，直接依赖版本见 `pyproject.toml`，当前安装解析版本见 `requirements.lock.txt`。正式模型与大规模索引在后续接入阶段确定。

| 层次 | 初始方案 | 用途 |
|---|---|---|
| 语言与包管理 | Python 3.10+、`pyproject.toml`、独立 `.venv` | 兼容当前 Python 3.10.14 环境 |
| 数据模型 | Pydantic | 输入输出校验和结构化模型响应解析 |
| 配置 | YAML + 环境变量 | 实验参数与密钥分离 |
| 命令行 | Typer | 数据处理、索引、推理和评价入口 |
| LLM | 当前 MockLLM，后续替换真实服务 | 统一规划、生成、mock 评估和优化调用 |
| Dense Retrieval | 当前 HashEmbedding + NumPy 精确检索 | 仅用于开发；后续替换语义模型和正式索引 |
| Sparse Retrieval | `rank-bm25` | 关键词检索 |
| Reranker | 后续接入轻量 Cross-Encoder | 当前保留接口，默认关闭 |
| Prompt | Jinja2 模板 | 统一维护不同智能体的提示词 |
| 测试与静态检查 | 测试暂缓；预留 Ruff 开发依赖 | 后续补充测试和检查流程 |
| 运行产物 | JSONL、JSON、CSV | 轨迹、预测、配置和结果汇总 |

当前离线流程仍默认使用 mock；成员三已加入 DeepSeek API 适配器和配置样例，但尚未进行带真实密钥的调用或正式质量实验。Embedding、模型权限和预算仍需全组确认。当前机器为 RTX 3050 6GB，离线运行不使用 GPU。模型调用统一经过适配器，后续接入时更新契约与实验记录。

## 4. 代码目录规划

以下包含已实现模块与后续扩展规划。数据读取、配置、临时检索、mock 规划生成、质量策略和循环编排已有实现；正式重排序与离线评价仍为扩展入口，`tests/` 暂未创建。

```text
HKUST-26fall-nlp-/
├── README.md                         # 项目说明、架构和进度计划
├── pyproject.toml                    # 依赖、包配置、开发工具
├── requirements.lock.txt             # 当前环境解析出的依赖版本
├── .env.example                      # 模型服务地址和密钥变量示例
├── .gitignore
│
├── configs/
│   ├── default.yaml                  # 完整系统默认配置
│   ├── baseline.yaml                 # Single-RAG 覆盖配置
│   ├── d3.yaml                       # 开启离线评估、优化与转接
│   └── experiments.yaml              # E0–E7 和补充消融覆盖配置
│
├── data/
│   ├── raw/                          # 原始 MultiDoc2Dial
│   ├── processed/                    # 标准化 passages 和对话样本
│   ├── samples/                      # 共享 mini dataset，包含离线标注
│   └── scenarios/                    # 显式 mock 服务脚本，不含数据集答案
│
├── artifacts/
│   └── indexes/                      # Dense / BM25 索引
│
├── src/
│   └── customer_service_rag/
│       ├── __init__.py
│       ├── __main__.py               # python -m customer_service_rag
│       ├── cli.py                    # config / doctor / preprocess / index / demo 等
│       ├── config.py                 # 配置加载、合并和校验
│       ├── schemas.py                # 统一数据模型
│       ├── interfaces.py             # 模块 Protocol 接口
│       │
│       ├── data/
│       │   ├── io.py                 # 共享 JSONL 读取与写入
│       │   ├── multidoc2dial.py      # 原始数据读取
│       │   ├── preprocessing.py     # 对话样本构造与标准化
│       │   └── chunking.py           # passage 切分及 grounding 映射
│       │
│       ├── models/
│       │   ├── llm.py                # LLM 调用、输出解析与用量记录
│       │   ├── mock_logic.py         # 离线词面评估与优化规则
│       │   └── embeddings.py         # Embedding 模型适配
│       │
│       ├── planning/
│       │   ├── planner.py            # 对话理解与查询规划入口
│       │   ├── rewriting.py          # 查询改写
│       │   ├── routing.py            # 意图和领域识别
│       │   └── decomposition.py      # 子问题分解
│       │
│       ├── retrieval/
│       │   ├── indexing.py           # 索引构建、保存与加载
│       │   ├── dense.py              # 向量检索
│       │   ├── bm25.py               # 关键词检索
│       │   ├── parallel.py           # 多检索器、多查询并行执行
│       │   └── pipeline.py           # 检索、去重、融合和重排序入口
│       │
│       ├── fusion/
│       │   ├── deduplication.py      # passage 去重与来源聚合
│       │   └── rrf.py                # Reciprocal Rank Fusion
│       │
│       ├── reranking/
│       │   └── reranker.py           # 候选证据重排序
│       │
│       ├── generation/
│       │   ├── generator.py          # 基于证据生成客服回复
│       │   └── citations.py          # 引用解析和有效性校验
│       │
│       ├── quality/
│       │   ├── evaluator.py          # 在线答案质量评估
│       │   └── policy.py             # pass / retry / clarify / escalate
│       │
│       ├── optimization/
│       │   └── optimizer.py          # 将反馈转为结构化优化动作
│       │
│       ├── orchestration/
│       │   ├── orchestrator.py       # 主流程与评估—优化循环
│       │   ├── loop.py               # 有界循环与组件注入
│       │   ├── actions.py            # 优化动作校验与执行
│       │   ├── scenarios.py          # 可控 mock 场景加载与服务替身
│       │   ├── state.py              # 单次请求的运行状态
│       │   ├── budget.py             # 重试、延迟与 Token 预算
│       │   └── escalation.py         # 人工转接上下文打包
│       │
│       ├── evaluation/
│       │   ├── retrieval_metrics.py  # Recall@k 等检索指标
│       │   ├── response_metrics.py   # F1、EM、BLEU 等回答指标
│       │   ├── grounding_metrics.py  # 引用、依据正确率和幻觉评价
│       │   └── runner.py             # 离线评价入口
│       │
│       ├── experiments/
│       │   ├── runner.py             # 批量执行实验配置
│       │   └── reporting.py          # 汇总结果表
│       │
│       ├── observability/
│       │   └── tracing.py            # 每轮输入输出、耗时和 Token 日志
│       │
│       └── prompts/
│           ├── planning.jinja2
│           ├── generation.jinja2
│           ├── evaluation.jinja2
│           └── optimization.jinja2
│
├── tests/                            # 后续规划，本轮暂不创建
│   ├── fixtures/                     # 模拟对话、证据和模型输出
│   ├── unit/                         # 数据映射、RRF、循环控制等测试
│   ├── contracts/                    # 模块输入输出兼容性测试
│   └── integration/                  # mini dataset 端到端测试
│
├── results/                          # 每次运行独立保存配置、预测和日志
├── reports/                          # 实验分析与成功、失败案例
└── docs/
    ├── architecture.md               # 详细架构说明
    ├── data_contracts.md             # 字段、状态和接口约定
    ├── contracts.schema.json         # 已导出的机器可读契约
    ├── progress.md                   # D1–D3 交付与后续待办
    ├── d3_walkthrough.md             # 场景、停止原因与实际运行记录
    └── experiment_protocol.md        # 数据划分、指标和实验设置
```

Python 子包已经补充 `__init__.py`。大型数据、索引、模型缓存、运行结果和真实 `.env` 已通过 `.gitignore` 排除；当前共享样例全部由项目编写，可纳入版本管理。

## 5. 数据模型与模块接口

### 5.1 统一数据模型

| 对象 | 主要字段与职责 |
|---|---|
| `DialogueInput` | `conversation_id`、当前问题、历史消息；只包含当前用户轮次之前可见的信息 |
| `QueryPlan` | 对话输入、`rewritten_query`、`intent`、`domain`、`sub_queries` |
| `Passage` | `doc_id`、`passage_id`、标题、正文、领域、原始 span / 字符偏移映射 |
| `RetrievalHit` | Passage、检索器来源、查询分支、原始分数与原始排名 |
| `RetrievalOptions` | 启用分支、各分支 top-k、融合候选数、重排序保留数 |
| `RetrievalResult` | 最终证据、各分支命中结果、融合分数、重排序分数、耗时 |
| `GeneratedResponse` | `answer`、`citations`、`retrieval_round`、`status` |
| `EvaluationFeedback` | 相关性、充分性、依据支持度、对话一致性评分，以及决策、缺失信息和建议动作 |
| `OptimizationAction` | 动作类型、更新后的查询或参数；支持重新检索和仅重新生成 |
| `RunState` | 当前计划、证据、候选回答、评估记录、迭代次数、预算和已见证据 ID |
| `RunResult` | 最终回复、最终状态、转接信息、运行轨迹、耗时和 Token 用量 |
| `EvaluationExample` | 对话输入、标准回复、标准 grounding、可替代证据分组；仅供离线评价使用 |

数据约定：

- Passage 用 `doc_id:passage_id` 作为稳定引用键，切分算法与参数需要记录版本。
- 去重时保留各检索器和查询分支的命中记录；RRF 使用各分支原始排名，不能在融合前丢失这些信息。
- 引用有效性检查验证引用是否来自当前提供的证据；事实是否获得支持由依据评价另行判断。
- 评分统一为 `[0, 1]`；通过阈值写入配置，并在验证集上确定。
- 评估决策统一为 `pass / retry / clarify / escalate`。
- 最终回复状态统一为 `answered / clarification_needed / insufficient_evidence / escalated / error`，与评估决策区分。
- 数据集标准回复和 grounding 不传给在线 Planner、Retriever、Generator 或 Evaluator，避免评价信息泄漏。
- 对话历史不包含当前待预测的客服回复和未来轮次。按官方可用划分构造样本，不随机打散同一对话的轮次。

### 5.2 核心接口

以下接口已在 `interfaces.py` 固定；字段定义见 `schemas.py`，可通过 `cs-rag schema` 导出 JSON Schema：

```python
class Planner(Protocol):
    async def plan(self, dialogue: DialogueInput) -> QueryPlan: ...


class Retriever(Protocol):
    async def retrieve(
        self, query: QueryPlan, options: RetrievalOptions
    ) -> RetrievalResult: ...


class Generator(Protocol):
    async def generate(
        self, query: QueryPlan, passages: list[Passage]
    ) -> GeneratedResponse: ...


class Evaluator(Protocol):
    async def evaluate(
        self,
        query: QueryPlan,
        passages: list[Passage],
        response: GeneratedResponse,
    ) -> EvaluationFeedback: ...


class Optimizer(Protocol):
    async def optimize(
        self, state: RunState, feedback: EvaluationFeedback
    ) -> OptimizationAction: ...


class Orchestrator(Protocol):
    async def run(self, dialogue: DialogueInput) -> RunResult: ...
```

模块通过接口和依赖注入连接，不直接访问其他模块的内部变量。通过 mock Planner、Retriever、Generator 和 Evaluator 可以在真实模型尚未接通时验证主流程。

### 5.3 在线与离线评价的边界

- `quality/`：运行时检查当前答案，不能查看标准答案。Evaluator 提供评分与诊断，Policy 结合阈值和预算形成可执行决策。
- `evaluation/`：实验后对照标准答案统计指标。模型判分时记录模型、Prompt 和判分结果，并人工抽查。

## 6. 四人分工与协作边界

成员编号沿用任务分配文档，实际姓名和负责人后续补充。

| 成员 | 主要职责 | 负责目录或文件 | 核心交付 |
|---|---|---|---|
| 成员一 | 数据、Single-RAG、离线评价 | `data/`、`evaluation/`、`experiments/`、`baseline.yaml` | 标准化数据、mini dataset、基线、指标与总体结果表 |
| 成员二 | 并行检索、融合、重排序 | `retrieval/`、`fusion/`、`reranking/`、`models/embeddings.py` | 索引、Dense、BM25、RRF、Reranker、检索消融 |
| 成员三 | 对话理解与回答生成 | `planning/`、`generation/`、规划和生成 Prompt、`models/llm.py` | Planner、Generator、模型调用适配器、上下文消融 |
| 成员四 | 质量检查、循环与系统集成 | `quality/`、`optimization/`、`orchestration/`、`observability/`、评估和优化 Prompt | Evaluator、Optimizer、预算控制、转接与完整系统 |
| 全组共同确定 | 共享契约与开发规范 | `schemas.py`、`interfaces.py`、`config.py`、`cli.py`、共享样例与实验协议 | 一致的字段、运行入口、配置与验收口径 |

协作约定：

1. D1 指定共享文件维护人，接口变更先更新契约，再同步使用方。
2. 各成员使用独立分支：`data-baseline`、`parallel-retrieval`、`dialogue-generation`、`evaluator-optimizer`。
3. Single-RAG 复用 Dense Retriever、Generator 和公共运行组件；成员一负责确定基线配置与评价，成员二、三提供对应实现，前期可用 mock 联调。
4. 公开接口需要契约测试；合并前通过相关检查和 mini dataset 验证，并附运行方法。
5. 每天同步完成项、接口变化、阻塞项和下一步；不把某人的模块内部实现作为他人的依赖。

## 7. 项目进度计划

### 7.1 排期原则与当前状态

采用 D1–D14 的两周计划，D1 表示正式开发启动日，不预设具体日历日期。排期以四人可并行开发、D1 能确定模型与运行资源为前提。每个阶段完成后按验收条件更新状态，计划日期不代表实际完成情况。

- [x] 阅读项目方案与任务分配文档。
- [x] 形成架构、模块边界、接口草案和进度计划。
- [x] 固定离线模型、当前设备、统一配置与自编样例使用边界。
- [x] 创建代码骨架、共享数据模型、临时索引和单次 mock 流程（D2）。
- [ ] 确定正式模型并核验官方数据许可、版本与划分。
- [x] 完成 mock 评估—优化多路径循环（D3），12 个控制场景可演示。
- [ ] 完成数据、Single-RAG 与独立模块。
- [ ] 完成完整系统集成与异常处理。
- [ ] 完成基线、消融实验和报告。

### 7.2 逐日计划

| 时间 | 工作重点 | 主要负责人 | 当日交付与验收条件 |
|---|---|---|---|
| D1 | 统一接口、技术选型、数据与实验规范 | 全组 | 已固定离线模型、字段、状态、设备、配置、指标口径与默认维护人；提供 16 条共享样例，正式模型接入边界另行记录 |
| D2 | 建立代码骨架并启动各模块 | 全组按目录分工 | 包可安装；配置可加载；共享模型可序列化；数据读取、临时索引、模型适配器和状态机分别启动 |
| D3 | 打通 mock 端到端流程 | 成员四牵头，全组配合 | 已完成评估、动作执行与受控循环；12 个场景及 16 条普通样例可运行，输出逐轮日志 |
| D4 | 首批真实数据与核心组件接入 | 成员一、二、三；成员四继续循环控制 | 生成 passages 和对话样本；Dense、BM25 可独立检索；真实模型完成规划和生成样例 |
| D5 | 完成主要算法与指标 | 全组并行 | RRF、Reranker、引用检查、在线评分、优化动作和离线指标具备可测试实现 |
| D6 | 独立模块验收与基线首跑 | 全组 | 各接口测试通过；mini dataset 上完成真实 Single-RAG 首跑并保存预测、检索结果、耗时和用量 |
| D7 | 使用正式语料建立索引并连接主链路 | 成员二、三、四；成员一提供数据 | Planner→Retriever→Generator 接通；证据 ID 和引用能映射回原文，运行配置与索引版本可追溯 |
| D8 | 接入 Evaluator–Optimizer Loop | 成员四牵头 | 优化动作能改变查询或检索参数；支持仅重新生成；循环次数和 Token / 延迟预算受控 |
| D9 | 完整系统集成验收 | 全组 | mini dataset 覆盖正常回答、缺证据、无新证据、超预算、模型异常、澄清和转接；接口与参数冻结 |
| D10 | 正式运行基线与检索消融 | 成员一、二牵头 | 在固定评价样本上运行 E0–E3；保存完整配置、逐样本预测、轨迹和指标 |
| D11 | 运行完整系统与循环消融 | 成员三、四牵头 | 运行 E4–E7；记录优化成功率、额外成本和最终状态，并完成上下文相关补充实验 |
| D12 | 复查实验与分析案例 | 全组 | 修复有依据的实验问题，重跑受影响配置；整理成功和失败案例；确认结果可在同配置下复核 |
| D13 | 撰写报告与制作图表 | 全组按章节分工 | 完成数据、方法、总体结果、消融和错误分析初稿；所有表格能追溯到运行结果 |
| D14 | 最终复现、演示与交付 | 全组 | 在干净环境按 README 复现 mini 流程；校对报告、运行说明和实验产物，完成演示 |

### 7.3 阶段里程碑

| 里程碑 | 目标时间 | 必须满足的验收条件 |
|---|---|---|
| M0：契约冻结 | D1 | 已固定离线开发数据模型、模块接口、状态、模型与评价协议；正式实验配置后续冻结 |
| M1：mock 闭环 | D3 | 不依赖真实索引和模型即可验证主要控制路径 |
| M2：模块可用 | D6 | 独立模块通过契约测试；mini Single-RAG 可运行 |
| M3：完整系统可用 | D9 | 真实数据端到端接通，重试、预算、澄清、转接和异常路径可验证 |
| M4：核心实验完成 | D12 | E0–E7 结果、配置、日志、成本和案例齐全 |
| M5：项目交付 | D14 | 代码、运行说明、实验结果、报告和演示可复核 |

### 7.4 依赖与范围管理

关键依赖链为：统一契约 → 标准化 passages → 正式索引 → 真实端到端流程 → 固定实验设置 → 实验结果 → 报告。

- 正式数据准备期间，成员二至四使用共享样例和 mock 独立开发。
- 先完成“单次检索与生成”，再接入“评估与优化”，每一步都保留可运行版本。
- 如果资源不足，优先缩减调参规模和可选实验；固定并公开最终评价样本范围，所有系统使用同一范围。
- 保留 MultiDoc2Dial、Single-RAG、Dense + BM25 + RRF + Reranker、查询改写、受控优化循环和核心消融。
- HotpotQA、第三检索分支、网页界面与真实人工客服平台集成安排在核心验收之后。

## 8. 实验设计与评价指标

### 8.1 核心消融实验

所有配置复用同一套流水线，通过配置开关实现，避免维护多份相似代码。

| 编号 | 配置 | 主要验证问题 |
|---|---|---|
| E0 | 当前问题 + Dense + 单次生成 | 从零建立 Single-RAG 基线 |
| E1 | E0 + BM25 并行检索 | 多路检索是否提高覆盖率 |
| E2 | E1 + RRF | 排名融合是否提高证据质量 |
| E3 | E2 + Reranker | 重排序是否改善最终证据 |
| E4 | E3 + 历史对话驱动的 Query Rewriting | 上下文处理是否改善检索与回答 |
| E5 | E4 + Evaluator / Policy，无优化重试 | 是否能识别并拦截不可靠回答 |
| E6 | E5 + Optimizer Loop | 重试是否修复证据或回答问题 |
| E7 | E6 + Human Escalation | 是否能为无法可靠回答的问题提供完整转接信息 |

实验口径：

- E1 关闭 RRF 时，采用固定的交替合并与去重规则；使用相同最终证据数量和上下文长度预算。
- E5 不合格时可返回澄清或信息不足；保留候选回答以评价识别能力，但不执行优化。
- E6 到达上限仍无法作答时返回 `insufficient_evidence`；E7 在需要人工处理时生成 `escalated` 转接包。
- E4–E7 的主实验先固定关闭子问题分解，单独做补充消融，避免多个变化混在一起。
- 补充实验比较：仅当前问题、问题加完整历史、查询改写、查询改写加子问题分解。
- E7 的实验可评价转接决策和信息完整性；真实人工处理效果需要另行接入平台并收集数据。

### 8.2 评价指标

| 类别 | 指标 | 计算与记录要求 |
|---|---|---|
| 检索 | Recall@1 / @5 / @10 | 保留 grounding 到 passage 的映射；区分命中任一证据与完整证据覆盖 |
| 回答 | Token F1、EM；可选 BLEU | 固定文本归一化、分词和多参考答案处理规则 |
| 依据 | 引用有效率、依据正确率 | 分开统计引用 ID 有效性与事实获得证据支持的程度 |
| 可靠性 | 幻觉率、对话一致性 | 固定判分标准；模型判分配合人工抽查 |
| 任务结果 | 作答率、澄清率、信息不足率、转人工比例 | 同时报告成功回答的质量与全样本结果，避免拒答改变分母造成误读 |
| 任务完成 | 任务完成率 | MultiDoc2Dial 中先明确可观测代理指标；没有标注依据时不宣称测得真实业务完成率 |
| 效率 | 平均及 P95 响应时间、平均检索次数 | 区分端到端耗时与各阶段耗时，记录硬件和并发条件 |
| 成本 | 输入 / 输出 Token、总模型调用次数；可选金额估算 | 覆盖 Planner、Generator、Evaluator、Optimizer 的全部调用 |

复现要求：固定数据版本、可用官方划分、passage 切分参数、索引、模型、Prompt、随机种子和生成参数。在验证集调参，最终评价集不参与阈值和 Prompt 调整。若官方测试集标准答案不可获得，应明确使用验证集留出部分等替代协议，不把无标签测试集写成已完成的有监督评价。

## 9. 配置设计

当前 `configs/default.yaml` 是可运行的离线配置；完整字段请执行 `cs-rag config` 查看。下方列出主要字段。评分阈值仍是开发初值，正式实验前需要在验证集确认。

```yaml
seed: 42

planning:
  use_history: true
  rewrite: true
  decompose: false

retrieval:
  branches: [dense, bm25]
  per_branch_top_k: 10
  fusion: rrf
  rrf_k: 60
  candidate_top_k: 10
  rerank: false
  final_top_k: 3

quality:
  enabled: false
  # 单次模式默认关闭；d3.yaml 开启 mock 评估与循环

optimization:
  enabled: false
  max_retries: 2
  stop_if_no_new_evidence: true

budget:
  max_latency_seconds: 30
  max_total_tokens: 12000

escalation:
  enabled: false
```

配置合并顺序为默认配置 → 指定配置 → 实验基础配置与 profile（如指定）→ 环境变量 → 显式命令行覆盖。相对路径以项目根目录为准，系统环境变量优先于 `.env`。每次运行保存最终展开配置。D3 已实现循环预算检查、请求超时及 mock 完成前的估算 Token 预留。正式模型需要另行校准用量估算。

`default.yaml` 和 `baseline.yaml` 继续提供单次模式；`d3.yaml` 开启 quality、optimization 和 escalation，保留 rerank=false。`--scenario` 自动采用 D3 配置并应用可见的场景覆盖，配置快照会记录最终值。

## 10. 命令行入口与实现状态

在项目虚拟环境中可使用 `cs-rag`，也可执行 `python -m customer_service_rag`。Windows 无需激活环境即可使用快速开始中的完整可执行文件路径。

```powershell
# 安装依赖
python -m pip install -c requirements.lock.txt -e .

# 环境、配置与契约
cs-rag doctor
cs-rag config --config configs/baseline.yaml
cs-rag config --experiment E4
cs-rag schema --output docs/contracts.schema.json
cs-rag samples

# 处理数据并准备 mini dataset
cs-rag preprocess --config configs/default.yaml

# 建立检索索引
cs-rag index --config configs/default.yaml

# 临时索引检索和单次 mock 演示
cs-rag retrieve --query "Which form replaces a lost driving license?"
cs-rag demo --sample-id mini_005

# 在 mini dataset 上运行基线配置（仍为 hash/mock，不是正式实验）
cs-rag run --config configs/baseline.yaml --split mini

# 运行当前单次离线流程
cs-rag run --config configs/default.yaml --split mini

# D3 普通规则与可控场景
cs-rag run --config configs/d3.yaml --split mini
cs-rag scenarios
cs-rag demo --scenario all
cs-rag demo --scenario retry-retrieve
cs-rag demo --config configs/d3.yaml --sample-id mini_013
```

`inspect-data --documents <本地文档JSON> --dialogues <本地对话JSON>` 已提供官方容器读取入口，但尚未在官方数据上核验。`chat`、`evaluate`、`experiment` 以及真实模型质量评价将在后续实现；E3–E7 仍缺少正式 Reranker 等模块。普通样例不会读取场景脚本，场景模式会显式标记 `mock_scenario`。本轮不执行测试套件。

## 11. 日志、测试与交付标准

### 11.1 单次运行产物

当前已保存 config、metadata、result、predictions、逐轮 retrieval、feedback、actions、candidates 和 traces；场景模式额外保存 scenario.json。metrics 与 summary 在正式离线评价实现后生成。

```text
results/<run_id>/
├── config.yaml               # 展开后的配置，排除密钥
├── metadata.json             # 代码版本、数据/索引/模型版本、环境信息
├── result.json               # 当前单次运行的完整结构化对象
├── predictions.jsonl         # 每个样本的最终回答与状态
├── retrieval.jsonl           # 分支检索、融合与重排序结果
├── traces.jsonl              # 各轮评分、动作、耗时、Token、停止原因
├── feedback.jsonl            # 各轮 Policy 后的评估反馈
├── actions.jsonl             # 校验后实际执行的动作
├── candidates.jsonl          # 每个初步答案，包括被拦截的草稿
├── scenario.json             # 仅场景模式：模拟服务脚本快照
├── metrics.json              # 后续：汇总指标
└── summary.csv               # 后续：便于比较的结果表
```

每个样本使用稳定的 `sample_id`，每次运行使用独立的 `run_id`。Token 用量无法从服务端获得时，明确标记为估算或缺失，不记为零。

### 11.2 关键测试

以下保留为后续开发清单；本轮按用户要求暂缓，不创建或执行测试套件。

- 数据：历史截断正确、无未来回复泄漏、grounding 可映射回 passage。
- 检索：去重保留来源、RRF 使用原始排名、top-k 与分支开关生效。
- 生成：结构化输出解析、引用 ID 校验、证据不足时的回复路径。
- 控制：通过、重试、仅重新生成、无新证据、重试上限、超时和 Token 上限。
- 异常：模型输出不可解析、服务调用失败、空检索结果、索引与语料版本不一致。
- 集成：mock 流程不依赖外部服务；真实模型 mini 流程单独运行并记录模型配置。

### 11.3 最终验收清单

- [ ] 干净环境能够按说明安装并运行 mini dataset。
- [ ] 数据处理与索引构建可重复执行，样本与引用可追溯。
- [ ] Single-RAG 与完整系统均能通过统一入口运行。
- [ ] 完整系统支持查询规划、并行检索、RRF、重排序、引用生成与质量检查。
- [ ] 优化循环受重试、证据变化、延迟和 Token 预算约束。
- [ ] 澄清与人工转接输出具有明确原因和可用上下文。
- [ ] E0–E7 及核心补充消融具有可追溯配置、预测、日志和指标。
- [ ] 报告说明效果提升、额外成本、失败案例与实验限制。
- [ ] 课程报告、图表和演示材料与最终代码及实验结果一致。

项目最终需要回答：并行检索是否提高证据召回率，评估—优化循环是否减少无依据回答，以及这些提升是否值得额外的延迟与计算成本。
