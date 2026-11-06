# Part 2：MultiDoc2Dial 混合证据检索

本模块接收上游 Query Planner 生成的查询，通过 Dense、BM25、RRF 和可选 Cross-Encoder 重排，从 MultiDoc2Dial 文档中检索证据，并将排好序的 `evidence` 交给下游 Response Generator。

```text
Query Planner
  → query / sub_queries / domain
  → Part 2 Hybrid Retrieval
  → evidence
  → Response Generator / Evaluator
```

## 目录速览

| 文件夹/文件 | 存放内容 |
| --- | --- |
| `retrieval/` | Part 2 核心代码，包括 BM25、Dense、RRF、重排、Pipeline 和 LangGraph 节点 |
| `scripts/` | 构建语料与索引、运行检索、下载模型和执行评估的命令行入口 |
| `examples/` | 上游输入 Part 2、下游读取 evidence 的最小接入示例 |
| `integration_demo/` | Part 1 → Part 2 → Part 3 的完整串联演示 |
| `tests/` | 单元测试和上下游接口契约测试 |
| `experiments/` | BM25、Embedding、召回和重排等优化实验脚本 |
| `docs/` | 实施手册、测试说明以及召回与重排优化记录 |
| `data/` | 放置 MultiDoc2Dial 原始数据；GitHub 中默认不提交数据文件 |
| `artifacts/` | 构建生成的 passage、BM25/FAISS 索引和评估结果；默认不提交大型文件 |
| `requirements.txt` | 项目直接依赖及固定版本 |
| `README.md` | 项目运行、输入输出和小组接入说明 |

## 1. 快速运行

### 使用当前已有环境和索引

```bash
cd /Users/leo/Desktop/hkust/5018/project/part2_retrieval_release
source ../.venv/bin/activate

export HF_HOME="../.model_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
```

运行默认 RRF 检索：

```bash
python -m scripts.run_retrieval \
  "How do I replace a lost driver's license?" \
  --domain dmv \
  --top-k 3 \
  --mode rrf \
  --artifact-dir ../artifacts \
  --device cpu
```

运行 Cross-Encoder 重排：

```bash
python -m scripts.run_retrieval \
  "How do I replace a lost driver's license?" \
  --domain dmv \
  --top-k 3 \
  --mode full \
  --artifact-dir ../artifacts \
  --device cpu
```

运行上下游接口示例：

```bash
python -m examples.upstream_downstream_example \
  --artifact-dir ../artifacts
```

运行测试：

```bash
pytest -q
```

当前应为 `60 passed`。可能出现一条 LangGraph 依赖弃用警告，不影响运行。

## 2. 关键运行参数

| 参数 | 含义 | 可选值/示例 |
| --- | --- | --- |
| 查询字符串 | 用户问题 | `How do I replace...` |
| `--domain` | 限制检索领域；不传则全局搜索 | `dmv`、`ssa`、`va`、`studentaid` |
| `--top-k` | 最终返回的 evidence 数量 | 1–20，常用 3 或 6 |
| `--mode` | 检索模式 | `bm25`、`dense`、`rrf`、`full` |
| `--artifact-dir` | passage 与索引所在目录 | 当前项目使用 `../artifacts` |
| `--device` | 模型运行设备 | 当前建议 `cpu` |

模式区别：

| mode | 执行内容 | 说明 |
| --- | --- | --- |
| `bm25` | 仅关键词召回 | 擅长精确术语和文档编号 |
| `dense` | 仅向量召回 | 擅长语义和同义表达 |
| `rrf` | Dense + BM25 + RRF | 默认模式，速度和效果较均衡 |
| `full` | RRF + Cross-Encoder | Top 排序更好，但 CPU 延迟约为 RRF 的 10 倍 |

## 3. 输入接口

Python 入口接收 `RetrievalRequest`：

```python
from retrieval import RetrievalRequest

request = RetrievalRequest(
    query="How can I replace a lost driver's license?",
    sub_queries=["lost license replacement documents"],
    domain="dmv",
    top_k=6,
    trace_id="dialog-12:turn-4",
)
```

| 字段 | 类型 | 必填 | 含义 |
| --- | --- | --- | --- |
| `query` | `str` | 是 | 主查询，不能为空 |
| `sub_queries` | `list[str]` | 否 | 上游生成的补充查询，最多采用 4 条 |
| `domain` | `str \| None` | 否 | 检索领域，`None` 表示全局搜索 |
| `top_k` | `int` | 否 | 最终 evidence 数量，范围 1–20 |
| `trace_id` | `str \| None` | 否 | 对话或请求追踪 ID |

父 LangGraph state 使用：

```python
{
    "query": "...",
    "sub_queries": ["..."],
    "domain": "dmv",
    "retrieval_top_k": 6,
    "trace_id": "dialog-12:turn-4"
}
```

其中只有 `query` 必须存在。Graph state 使用 `retrieval_top_k`，Python Request 使用 `top_k`。

## 4. 输出接口

Python Pipeline 返回 `list[Evidence]`；LangGraph node 返回：

```python
{"evidence": [evidence_1, evidence_2, ...]}
```

单条 evidence 示例：

```json
{
  "passage_id": "Replace license or permit#1_0::1",
  "doc_id": "Replace license or permit#1_0",
  "domain": "dmv",
  "title": "Replace license or permit#1",
  "section_title": "Replace license or permit[1]",
  "text": "You can replace your license or permit...",
  "span_ids": ["2", "3"],
  "source_retrievers": ["bm25", "dense"],
  "matched_query": "How do I replace a lost driver's license?",
  "dense_rank": 3,
  "bm25_rank": 6,
  "rrf_score": 0.0581,
  "rerank_score": 5.3132
}
```

| 字段 | 含义 |
| --- | --- |
| `passage_id` | evidence 唯一 ID，用于引用、去重和评估 |
| `doc_id`、`domain` | 来源文档与领域 |
| `title`、`section_title`、`text` | 下游生成回答使用的主要上下文 |
| `span_ids` | 对应原始 MultiDoc2Dial span，便于追溯 |
| `source_retrievers` | 哪些召回器找到该 passage |
| `matched_query` | 命中该 passage 的主查询或子查询 |
| `dense_rank`、`bm25_rank` | Dense/BM25 单路排名；未命中时为 `null` |
| `rrf_score` | RRF 融合分数 |
| `rerank_score` | Cross-Encoder 分数；RRF 模式下与融合分数相同 |

下游至少使用 `passage_id/title/section_title/text`，并保持 evidence 当前顺序。`full` 模式的最终顺序来自重排模型，不能再次按 `rrf_score` 排序。

## 5. 程序主链

```text
scripts/run_retrieval.py
  → RetrievalConfig.for_mode
  → HybridRetrievalPipeline.from_artifacts
  → pipeline.retrieve
      → 校验输入
      → Dense 与 BM25 并行召回
      → Reciprocal Rank Fusion
      → 可选 Cross-Encoder 重排
      → 截取 top_k
      → list[Evidence]
```

建议阅读顺序：

1. `retrieval/schemas.py`：输入输出结构；
2. `retrieval/config.py`：模式与参数；
3. `retrieval/pipeline.py`：完整调用链；
4. `retrieval/bm25.py`、`dense.py`：两路召回；
5. `retrieval/fusion.py`、`reranker.py`：融合与重排；
6. `retrieval/graph_node.py`：LangGraph 接入。

## 6. 接入父 LangGraph

Pipeline 应在应用启动时加载一次，不要在每个请求中重复加载模型：

```python
from retrieval import HybridRetrievalPipeline, RetrievalConfig
from retrieval.graph_node import make_retrieval_node

config = RetrievalConfig.for_mode("rrf", artifact_dir="artifacts")
pipeline = HybridRetrievalPipeline.from_artifacts(
    artifact_dir="artifacts",
    config=config,
    device="cpu",
)

retrieve_evidence = make_retrieval_node(pipeline)

builder.add_node("query_planner", query_planner)
builder.add_node("retrieve_evidence", retrieve_evidence)
builder.add_node("response_generator", response_generator)

builder.add_edge("query_planner", "retrieve_evidence")
builder.add_edge("retrieve_evidence", "response_generator")
```

检索节点只返回 `{"evidence": [...]}`，不会覆盖父 state 中其他模块的字段。

完整示例见 `examples/upstream_downstream_example.py`。`integration_demo/` 提供可运行的 Part 1 → Part 2 → Part 3 演示，但其中 Part 1 和 Part 3 只是可替换的规则示例。

## 7. 新环境从零构建

### 安装环境

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export HF_HOME="$PWD/.model_cache"
```

### 准备数据

从 [MultiDoc2Dial 数据说明页](https://doc2dial.github.io/multidoc2dial/data_readme.html) 获取数据，并放到：

```text
data/
├── multidoc2dial_doc.json
└── multidoc2dial_dial_validation.json
```

### 构建 passage

```bash
python -m scripts.build_corpus \
  --documents data/multidoc2dial_doc.json \
  --output artifacts/passages.jsonl
```

正常情况下生成约 13,421 条 passage。

### 构建 BM25 与 Dense 索引

```bash
python -m scripts.build_indexes \
  --artifact-dir artifacts \
  --device cpu
```

首次运行会下载 `BAAI/bge-base-en-v1.5`。如果下载较慢，可以使用：

```bash
python -m scripts.download_models \
  --cache-dir .model_cache \
  --mirror-first
```

索引构建完成后，运行命令中的目录应改为：

```bash
--artifact-dir artifacts
```

不要把数据集、模型缓存或大型索引提交到 GitHub，`.gitignore` 已经排除这些文件。

## 8. Validation 评估

```bash
python -m scripts.evaluate_retrieval \
  --documents data/multidoc2dial_doc.json \
  --dialogues data/multidoc2dial_dial_validation.json \
  --artifact-dir artifacts \
  --mode rrf \
  --query-mode history \
  --limit 80 \
  --balanced-by-domain \
  --output artifacts/evaluation_rrf_80.json
```

关键指标：

| 指标 | 含义 |
| --- | --- |
| `hit_rate@k` | Top-k 是否至少命中一个 Gold 的样本比例 |
| `recall@k` | 每个样本的全部 Gold 被覆盖的平均比例 |
| `user_gold_*` | 只评估 user turn Gold |
| `agent_gold_*` | 只评估 agent turn Gold |
| `mrr` | 第一个相关 passage 的倒数排名均值 |
| `mean/p50/p95_latency_ms` | 平均、中位数和长尾延迟 |

现有 400 条均衡样本结果：

| 模式 | Hit@10 | Recall@10 | MRR |
| --- | ---: | ---: | ---: |
| RRF | 55.50% | 49.42% | 0.3084 |
| guarded Full | 60.75% | 55.38% | 0.3997 |

详细优化记录位于 `docs/RETRIEVAL_OPTIMIZATION_LOG.md` 和 `docs/RERANKER_OPTIMIZATION_LOG.md`。

## 9. 常见问题

### 找不到 `retrieval` 模块

确认当前位于仓库根目录，并使用模块方式运行：

```bash
python -m examples.upstream_downstream_example --artifact-dir artifacts
```

### 提示缺少 passage 或 index

检查 `--artifact-dir` 是否正确，以及目录中是否存在 `passages.jsonl`、`bm25.pkl` 和 `faiss.index`。

### `rerank_score` 与 `rrf_score` 相同

这是因为使用了 `rrf` 模式。改成 `--mode full` 才会调用 Cross-Encoder。

### 模型无法下载

首次下载前不要设置 `HF_HUB_OFFLINE=1`。国内网络可以使用 `scripts.download_models --mirror-first`。

## 10. 模块责任边界

- 上游负责：`query`、可选 `sub_queries` 和 `domain`；
- Part 2 负责：输入校验、召回、融合、重排和 evidence 输出；
- 下游负责：按照 evidence 当前顺序生成答案和引用；
- Evaluator/Optimizer 负责：判断质量，必要时更新 query 并重新调用同一个检索节点。

其他模块不要直接调用 `DenseRetriever`、`BM25Retriever` 或读取 FAISS 行号，应只依赖 `RetrievalRequest`、`Evidence` 和 `make_retrieval_node`。
