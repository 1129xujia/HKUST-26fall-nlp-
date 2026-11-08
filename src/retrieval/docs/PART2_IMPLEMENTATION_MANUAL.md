> 2026-10-28 更新：本文保留原设计/源码导学结构；当前可执行命令、环境与验收结果以 `TESTING_GUIDE.md`、`retrieval/config.py` 和 `artifacts/acceptance/summary.json` 为准。新增独立 Demo 见 `integration_demo/README.md`。

> 2026-10-29 优化更新：下文的 Top-K、RRF、重排与去重数值是最初实施方案，不再代表当前默认值。默认入口是 BGE Dense Top 100 + BM25 Top 50 + RRF k=30，不加载 Cross-Encoder；显式 `full` 模式会使用 current-first user-history query 重排 Top 50。额外语义/每文档去重关闭。召回过程见 `RETRIEVAL_OPTIMIZATION_LOG.md`，重排过程见 `RERANKER_OPTIMIZATION_LOG.md`。

# Part 2：并行检索、RRF 融合与重排序操作手册

## 1. 你的任务边界

你负责的模块位于 Query Planning Agent 与 Response Generator 之间：

```text
Query Planning Agent
        |
        | RetrievalRequest
        v
+---------------------------------------------+
| 你的 Part 2                                  |
| Dense Retrieval + BM25 -> RRF               |
|              -> optional Rerank/Deduplicate |
+---------------------------------------------+
        |
        | list[Evidence]
        v
Response Generator / Evaluator-Optimizer Loop
```

你的模块只负责“寻找并排序证据”，不负责：

- 根据多轮对话重写查询；
- 生成最终客服答案；
- 判断答案是否需要重试或转人工；
- 在模块内部调用回答生成 LLM。

你的最终交付应让队友只需要调用一个稳定接口：

```python
evidence = pipeline.retrieve(request)
```

即可获得 5 至 8 条带来源、分数和引用信息的证据。

---

## 2. 完成标准

你的 Part 2 必须同时满足以下条件：

1. 能读取 `multidoc2dial/multidoc2dial_doc.json` 并建立 passage 库。
2. 能分别运行 Dense Retriever 和 BM25 Retriever。
3. 两个 Retriever 可以并行执行。
4. 能使用 passage ID 对两个 ranked list 做 RRF 融合。
5. 能用 Cross-Encoder 对融合后的候选重新排序。
6. 能去除完全重复、相邻重叠或语义高度重复的证据。
7. 对外提供固定的 Python 输入输出接口。
8. 能在 validation split 上计算 Recall@k、MRR、延迟和重复率。
9. Dense、BM25、RRF、RRF + Reranker 四种配置可以通过配置项切换。
10. 队友不需要知道 FAISS、BM25 或 reranker 的内部实现。

---

## 3. 推荐目录结构

在当前项目中建立：

```text
project/
├── multidoc2dial/
│   ├── multidoc2dial_doc.json
│   ├── multidoc2dial_dial_train.json
│   ├── multidoc2dial_dial_validation.json
│   └── multidoc2dial_dial_test.json
├── retrieval/
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   ├── corpus.py
│   ├── dense.py
│   ├── bm25.py
│   ├── fusion.py
│   ├── reranker.py
│   ├── deduplication.py
│   └── pipeline.py
├── scripts/
│   ├── build_corpus.py
│   ├── build_indexes.py
│   └── evaluate_retrieval.py
├── tests/
│   ├── test_corpus.py
│   ├── test_fusion.py
│   ├── test_pipeline.py
│   └── test_integration_contract.py
├── artifacts/
│   ├── passages.jsonl
│   ├── passage_embeddings.npy
│   ├── faiss.index
│   ├── bm25.pkl
│   └── index_metadata.json
├── requirements.txt
└── README.md
```

`artifacts/` 中的文件是可重新生成的索引产物。代码中不要写死当前电脑的绝对路径。

---

## 4. 环境准备

建议使用 Python 3.10 或更高版本。最小依赖：

```text
numpy
torch
transformers
sentence-transformers
faiss-cpu
rank-bm25
scikit-learn
tqdm
pytest
```

建议先把依赖固定在 `requirements.txt`，所有组员使用同一份依赖。

安装后先验证：

```bash
python -c "import faiss, torch, sentence_transformers, rank_bm25"
```

模型建议：

- Dense embedding：`BAAI/bge-base-en-v1.5`
- Reranker：`cross-encoder/ms-marco-MiniLM-L6-v2`

第一次运行需要下载模型。模型名必须放进配置文件，不要散落在业务代码中。

---

## 5. 第一步：先和队友冻结接口

在写 Retriever 前，先将以下接口发给负责 Part 1、生成器和 Evaluator 的队友确认。

### 5.1 输入对象

在 `retrieval/schemas.py` 中定义：

```python
from dataclasses import dataclass, field


@dataclass
class RetrievalRequest:
    query: str
    sub_queries: list[str] = field(default_factory=list)
    domain: str | None = None
    top_k: int = 6
    trace_id: str | None = None
```

字段含义：

- `query`：Part 1 产生的自包含主查询；不能为空。
- `sub_queries`：可选子查询；最多接收 3 个。
- `domain`：可选的 `dmv`、`ssa`、`va` 或 `studentaid`；为空时搜索全库。
- `top_k`：希望返回的最终证据数，默认 6。
- `trace_id`：用于串联一次请求及之后的优化重试。

### 5.2 输出对象

```python
@dataclass
class Evidence:
    passage_id: str
    doc_id: str
    domain: str
    title: str
    section_title: str
    text: str
    span_ids: list[str]
    source_retrievers: list[str]
    matched_query: str
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float = 0.0
```

不要只向队友返回 passage 文本。`doc_id`、`passage_id` 和 `span_ids` 是生成引用、评估 gold evidence 和定位错误所必需的。

### 5.3 唯一公共入口

```python
class HybridRetrievalPipeline:
    def retrieve(self, request: RetrievalRequest) -> list[Evidence]:
        ...
```

队友只能依赖这个公共入口，不应直接调用 `DenseRetriever` 或 `BM25Retriever`。

---

## 6. 第二步：构建 passage 语料库

### 6.1 不要立即按固定 token 粗暴切块

MultiDoc2Dial 已经在每篇文档的 `spans` 中提供：

- `id_sp`：gold span ID；
- `id_sec`：span 所属 section ID；
- `text_sec`：section 文本；
- `title`：section 标题；
- `parent_titles`：父级标题；
- `start_sec` 和 `end_sec`：section 在文档中的位置。

第一版应将唯一的 `(doc_id, id_sec)` 作为 passage。这样可以直接建立 passage 与 gold span 的映射。

### 6.2 Passage ID

使用稳定 ID：

```python
passage_id = f"{doc_id}::{id_sec}"
```

不要使用列表下标作为永久 ID，因为重新处理数据后下标可能改变。

### 6.3 Passage 记录

每行写入 `artifacts/passages.jsonl`：

```json
{
  "passage_id": "document-id::section-id",
  "doc_id": "document-id",
  "domain": "dmv",
  "title": "document title",
  "section_title": "section title",
  "parent_titles": ["parent title"],
  "text": "section text",
  "span_ids": ["23", "24", "25"],
  "start": 100,
  "end": 530
}
```

构建算法：

1. 遍历 `doc_data[domain][doc_id]`。
2. 遍历文档的 `spans`。
3. 按 `(doc_id, id_sec)` 聚合。
4. 对相同 section 的 `span_ids` 去重并排序。
5. 使用 `text_sec` 作为 passage 文本。
6. 丢弃空文本，但记录丢弃数量。
7. 如果 section 超过约 350 tokens，再进行子切块，同时保留它包含的 span ID。

索引文本建议拼接标题信息：

```python
index_text = "\n".join(
    x for x in [document_title, section_title, passage_text] if x
)
```

返回给生成器的 `text` 仍使用正文，不要把重复标题混进证据正文。

### 6.4 构建完成后的检查

必须确认：

- `passage_id` 无重复；
- 所有 passage 的 `text` 非空；
- 每条记录都能回溯到合法 `doc_id`；
- `span_ids` 不为空；
- 四个 domain 都有 passage；
- 同样输入重复构建两次时，passage ID 和顺序稳定。

---

## 7. 第三步：建立 Dense 索引

### 7.1 离线构建

对每条 passage 的 `index_text` 编码：

```python
embeddings = model.encode(
    index_texts,
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True,
)
```

使用 FAISS cosine 检索时，可对归一化向量使用 inner product：

```python
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype("float32"))
```

同时保存：

- `faiss.index`
- `passage_embeddings.npy`
- FAISS 行号到 `passage_id` 的映射
- embedding 模型名和版本信息

### 7.2 在线检索

每个 query 默认返回 Top 20。输出必须转换成统一内部格式：

```python
SearchHit(
    passage_id="...",
    retriever="dense",
    rank=1,
    raw_score=0.82,
    matched_query=query,
)
```

如果 request 指定了 domain，可以：

- 第一版先扩大候选后过滤 domain；
- 稳定版为每个 domain 建独立 FAISS index。

---

## 8. 第四步：建立 BM25 索引

BM25 必须与 Dense 使用完全相同的 passage 集合和 `passage_id`。

英文最小分词方式：

```python
import re


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())
```

不要删除数字、连字符词、缩写和政策编号。BM25 的主要作用正是找出精确名称、号码和政策术语。

每个查询默认返回 Top 20，并输出相同的 `SearchHit` 结构，只把 `retriever` 改为 `bm25`。

BM25 原始分数不能和 Dense 原始分数直接相加或平均。

---

## 9. 第五步：并行检索

对于一个主查询和若干子查询，形成：

```python
queries = [request.query] + request.sub_queries[:3]
```

每个 query 都运行 Dense 和 BM25。第一版可以使用线程池：

```python
from concurrent.futures import ThreadPoolExecutor


with ThreadPoolExecutor(max_workers=2) as pool:
    dense_future = pool.submit(dense.search, query, 20, domain)
    bm25_future = pool.submit(bm25.search, query, 20, domain)
    dense_hits = dense_future.result()
    bm25_hits = bm25_future.result()
```

最终 RRF 接收的 ranked lists 数量为：

```text
查询数量 × 2 个 Retriever
```

必须为每个结果保留 `matched_query`，否则后续无法解释某条证据是由哪个子查询找到的。

---

## 10. 第六步：RRF 融合

RRF 公式：

```text
RRF(d) = Σ 1 / (k + rank(d))
```

推荐初始参数：

```text
rrf_k = 60
dense_top_k = 20
bm25_top_k = 20
fused_top_k = 30
```

实现注意事项：

1. rank 从 1 开始。
2. 使用 `passage_id` 合并相同 passage。
3. 同一 passage 出现在多个列表中时累加分数。
4. 记录其来自 Dense、BM25 或两者。
5. 记录最好的 Dense/BM25 rank。
6. 分数相同时，使用稳定的 passage ID 作为最后排序键，保证测试可复现。

核心函数应保持纯函数形式：

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchHit]],
    rrf_k: int = 60,
) -> list[FusedHit]:
    ...
```

RRF 单元测试至少覆盖：

- 单列表排序不被破坏；
- 同一 passage 在两个列表出现时分数正确累加；
- 缺失某个 Retriever 结果时仍可运行；
- 空列表不会导致除零或崩溃；
- 多个子查询产生的列表可以正确融合。

---

## 11. 第七步：Cross-Encoder 重排序

将 RRF Top 30 传给 reranker：

```python
pairs = [(query, candidate.index_text) for candidate in candidates]
scores = reranker.predict(pairs)
```

如果有多个子查询，建议为每个候选计算：

```text
主查询分数 + 所有子查询分数
```

第一版最终分数可以取最大值：

```python
rerank_score = max(score(main_query), score(each_sub_query))
```

同时记录获得最大分数的 `matched_query`。

稳定版可以采用：

```text
0.7 × 主查询分数 + 0.3 × 最佳子查询分数
```

但必须通过 validation 实验决定，不要只凭直觉修改权重。

当 reranker 加载失败时，系统应退化为 RRF 排序，而不是让整个服务崩溃。

---

## 12. 第八步：去重和证据覆盖

按照 reranker 顺序逐条选择，直到得到 `request.top_k` 条证据。

按以下顺序过滤：

1. 相同 `passage_id`：直接去重。
2. 相同 `doc_id` 且文本规范化后相同：只保留高分项。
3. 相同文档中位置高度重叠：只保留高分项。
4. passage embedding cosine similarity 大于 0.90：视为语义重复。

为了避免所有结果都集中在同一 section，可以设置：

```text
同一个 doc_id 最多保留 3 条 passage
```

如果存在子查询，应优先保证每个有效子查询至少有一条高分证据，再用总排序填充剩余位置。

如果严格去重后不足 `top_k`，继续扫描后面的 RRF 候选，不要直接返回空位。

---

## 13. 第九步：组装完整 Pipeline

`retrieval/pipeline.py` 的流程应固定为：

```python
class HybridRetrievalPipeline:
    def retrieve(self, request: RetrievalRequest) -> list[Evidence]:
        self._validate_request(request)

        queries = self._prepare_queries(request)
        ranked_lists = self._parallel_search(
            queries=queries,
            domain=request.domain,
        )

        fused = reciprocal_rank_fusion(
            ranked_lists,
            rrf_k=self.config.rrf_k,
        )[: self.config.fused_top_k]

        reranked = self.reranker.rerank(
            main_query=request.query,
            sub_queries=request.sub_queries,
            candidates=fused,
        )

        selected = self.deduplicator.select(
            reranked,
            top_k=request.top_k,
        )

        return [self._to_evidence(item) for item in selected]
```

请求校验至少包括：

- query 不能是空字符串；
- sub-query 去除空值和重复值；
- sub-query 最多 3 个；
- domain 必须属于允许集合；
- top_k 必须位于 1 至 20；
- 所有字符串去除首尾空格。

---

## 14. 第十步：配置管理

不要把参数写死在各个模块中。集中定义配置：

```python
@dataclass
class RetrievalConfig:
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    dense_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    fused_top_k: int = 30
    final_top_k: int = 6
    dedup_similarity: float = 0.90
    max_sub_queries: int = 3
    use_dense: bool = True
    use_bm25: bool = True
    use_reranker: bool = True
```

索引元数据中保存构建时的 embedding 模型名。启动 pipeline 时，如果当前配置模型与索引模型不一致，应明确报错并要求重建索引。

---

## 15. 第十一步：构造检索评估集

开发阶段使用 train 的一小部分调试，正式比较使用 validation。不要在 validation 上训练模型，也不要使用 test 调参数。

### 15.1 查询

当 Part 1 尚未完成时，可以先用当前用户 utterance 作为查询基线。

正式集成后，使用 Part 1 输出的自包含 query；同时保留原始 utterance，便于比较 query rewriting 是否带来提升。

### 15.2 Gold evidence

对于一个用户请求，建议把以下 references 的并集作为 gold：

```text
当前用户 turn 的 references
+ 紧随其后的 agent answer 的 references
```

理由：用户 turn 的引用经常描述 precondition，而 agent turn 的引用经常描述 solution。只取其中一侧会低估检索效果。

映射方式：

```text
(doc_id, id_sp) -> 包含该 span_id 的 passage_id
```

这一评估定义必须和全组确认，并写入实验报告。若对话中不存在合法的下一条 agent turn，只使用当前 turn references，或将该样本标记为无法配对并单独统计。

### 15.3 指标

至少输出：

- Passage Recall@5
- Passage Recall@10
- Span Recall@5
- Span Recall@10
- MRR@10
- 平均查询延迟
- P50/P95 查询延迟
- 最终结果重复率
- 无结果请求比例

Recall 的定义必须明确是“找到任意 gold”还是“找到所有 gold”。本项目分别报告：

```text
Hit Rate：至少命中一个 gold 的请求比例
Recall：被检索到的 gold passage 数 / 全部 gold passage 数
User Gold Hit Rate / Recall：只评估 user turn 标注的 Gold
Agent Gold Hit Rate / Recall：只评估下一条 agent turn 标注的 Gold
```

总体指标使用两组 Gold 的去重并集。拆分指标只对相应 Gold 非空的样本取平均，并同时报告各自的样本数。

---

## 16. 第十二步：完成消融实验

运行以下四组：

| 实验 | Dense | BM25 | RRF | Reranker | Dedup |
|---|---:|---:|---:|---:|---:|
| A | 是 | 否 | 否 | 否 | 否 |
| B | 否 | 是 | 否 | 否 | 否 |
| C | 是 | 是 | 是 | 否 | 否 |
| D | 是 | 是 | 是 | 是 | 是 |

结果表至少包含：

```text
method, recall@5, recall@10, mrr@10,
coverage@5, mean_latency_ms, p95_latency_ms, duplicate_rate
```

同时保存每个样本的详细结果，方便定位：

- Dense 成功但 BM25 失败的语义改写案例；
- BM25 成功但 Dense 失败的编号或精确术语案例；
- RRF 提升排名的案例；
- reranker 把正确 passage 排到前面的案例；
- 去重误删重要证据的失败案例。

---

## 17. 集成协议

### 17.1 Part 1 接入你的模块

Part 1 只需要产生：

```python
request = RetrievalRequest(
    query=planned_query,
    sub_queries=sub_queries,
    domain=predicted_domain,
    top_k=6,
    trace_id=trace_id,
)
```

如果 domain 不确定，传 `None`，你的模块必须能够全库搜索。

### 17.2 你接入 Response Generator

返回的每条 Evidence 必须包含：

```text
[passage_id]
title
section_title
text
doc_id
span_ids
```

生成器应基于这些 ID 生成引用，例如 `[E1]`、`[E2]`。不要让生成器自己猜文档来源。

### 17.3 接入 Evaluator-Optimizer Loop

Optimizer 发起第二次检索时，仍使用同一接口。它可以：

- 修改 `query`；
- 添加缺失的 `sub_queries`；
- 移除错误的 domain 限制；
- 调高 `top_k`，但不能超过约定上限。

你的模块应保持无状态，或者将缓存与业务状态分离。一次失败的请求不应污染下一次请求。

### 17.4 错误处理

建议定义明确异常：

```python
class InvalidRetrievalRequest(ValueError): ...
class RetrievalIndexNotFound(RuntimeError): ...
class RetrievalModelMismatch(RuntimeError): ...
```

以下情况不要静默返回空列表：

- 索引文件不存在；
- passage 元数据与 FAISS 行数不一致；
- 索引使用的 embedding 模型与当前模型不一致。

正常的“没有高相关结果”可以返回空列表或低分结果，但必须通过日志或 metadata 区分于系统故障。

---

## 18. 日志和可观测性

每次调用至少记录：

```text
trace_id
query_count
domain
dense_latency_ms
bm25_latency_ms
fusion_latency_ms
rerank_latency_ms
total_latency_ms
dense_result_count
bm25_result_count
final_result_count
```

不要在默认日志中打印完整用户对话或整篇文档。调试模式下可以输出 passage ID 和分数。

---

## 19. 必须编写的测试

### 19.1 Corpus 测试

- 相同 `(doc_id, id_sec)` 被正确聚合；
- `span_ids` 没有丢失；
- passage ID 唯一；
- 空 section 被安全处理。

### 19.2 Retriever 测试

- Dense 和 BM25 都返回统一的 `SearchHit`；
- top_k 生效；
- domain filter 生效；
- 空 query 被拒绝；
- top_k 大于语料数时不会崩溃。

### 19.3 Fusion 测试

- RRF 公式正确；
- passage 重复出现时分数累加；
- 单 Retriever 故障时仍可退化运行；
- 排序可复现。

### 19.4 Integration Contract 测试

从真实 validation 数据取一个样本，完成：

```text
JSON 对话 -> RetrievalRequest -> pipeline.retrieve()
          -> list[Evidence] -> JSON 序列化
```

验证输出字段完整，并确保队友无需读取你的内部对象。

---

## 20. 交付给队友前的检查清单

### 代码

- [ ] `HybridRetrievalPipeline.retrieve()` 可以单独调用。
- [ ] 所有公共类型都有类型标注和 docstring。
- [ ] 路径通过参数或配置传入，没有个人电脑绝对路径。
- [ ] 模型与超参数集中配置。
- [ ] Dense/BM25 任意一个关闭时仍可运行。
- [ ] reranker 故障时可以退化到 RRF。
- [ ] 所有单元测试通过。

### 数据与索引

- [ ] passage ID 稳定且唯一。
- [ ] 每条 passage 保留 `doc_id` 和 `span_ids`。
- [ ] 索引元数据记录模型名、构建时间和 passage 数量。
- [ ] FAISS 行数与 passage 数量完全一致。
- [ ] 提供一条命令可以从原始 JSON 重建全部索引。

### 实验

- [ ] 已完成 Dense-only、BM25-only、RRF、RRF + reranker 对比。
- [ ] 已保存 validation 指标和逐样本结果。
- [ ] 已报告 Recall、MRR、延迟和重复率。
- [ ] 已记录至少 3 个成功案例和 3 个失败案例。

### 接入

- [ ] 已把 `RetrievalRequest` 和 `Evidence` schema 发给队友确认。
- [ ] 已给队友一个最小调用示例。
- [ ] 已确认生成器使用 `passage_id` 做引用。
- [ ] 已确认 Optimizer 重试仍调用同一个接口。
- [ ] 已进行一次真实的端到端联调。

---

## 21. 推荐执行顺序

### 第 1 天：接口和语料

1. 与队友确认输入输出 schema。
2. 实现 section-based passage 构建。
3. 输出 `passages.jsonl`。
4. 完成 corpus 单元测试。

### 第 2 天：双路检索

1. 建立 FAISS 索引。
2. 建立 BM25 索引。
3. 分别完成 Top 20 检索。
4. 在几个真实 query 上人工检查结果。

### 第 3 天：融合和重排序

1. 实现 RRF。
2. 完成 RRF 单元测试。
3. 接入 Cross-Encoder。
4. 实现 passage 去重和覆盖约束。

### 第 4 天：评估

1. 构造 validation retrieval samples。
2. 运行四组消融实验。
3. 输出 CSV/JSON 指标。
4. 分析成功和失败案例。

### 第 5 天：集成

1. 封装 `HybridRetrievalPipeline`。
2. 与 Part 1 的查询输出联调。
3. 与 Response Generator 的证据输入联调。
4. 测试 Optimizer 第二次调用。
5. 更新项目 README 和运行命令。

---

## 22. 最小可运行配置

如果时间紧，先交付以下版本：

```text
Passage             = MultiDoc2Dial 原生 section
Dense model         = BAAI/bge-base-en-v1.5
Dense top-k         = 20
BM25 top-k          = 20
RRF k               = 60
RRF candidates      = 30
Reranker            = ms-marco-MiniLM-L6-v2
Final evidence      = 6
Semantic dedup      = cosine similarity > 0.90
Max sub-queries     = 3
```

最小版本完成的标志不是“代码能够运行一次”，而是：

> 任意队友传入合法 `RetrievalRequest` 后，都能稳定得到可 JSON 序列化、可引用、可评估的 `list[Evidence]`；同时你能用 validation 指标证明融合检索相对于单一检索的效果和代价。

---

## 23. 最小调用示例

```python
from retrieval.config import RetrievalConfig
from retrieval.pipeline import HybridRetrievalPipeline
from retrieval.schemas import RetrievalRequest


config = RetrievalConfig()
pipeline = HybridRetrievalPipeline.from_artifacts(
    artifact_dir="artifacts",
    config=config,
)

request = RetrievalRequest(
    query="How can I update the address on my driver's license?",
    sub_queries=[
        "DMV address change deadline",
        "online DMV address change procedure",
    ],
    domain="dmv",
    top_k=6,
    trace_id="demo-001",
)

evidence = pipeline.retrieve(request)

for item in evidence:
    print(item.passage_id, item.rerank_score, item.title)
```

队友接入时，应只需要理解这个调用示例和两个 schema，而不需要阅读你内部的检索实现。
