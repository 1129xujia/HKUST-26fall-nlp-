# 项目验收与复现步骤

更新日期：2026-10-28。所有命令从本地项目根目录执行：

```bash
cd /Users/leo/Desktop/hkust/5018/project
```

## 1. 当前机器：建议先做完整验收

模型、数据和索引在本地保留后，验收可以离线运行。

```bash
source .venv_acceptance/bin/activate
export HF_HOME="$PWD/.model_cache"
export HF_HUB_OFFLINE=1
python -m pip --no-cache-dir check
python -m scripts.verify_project --full --limit 80 --device cpu
python -m integration_demo.render_report
```

预期依次出现依赖无冲突、单元测试通过、四种模式完成评估和 `Acceptance passed`。验收脚本不自动补模型、不偷偷降级：Full 模式若模型或索引缺失会直接失败。

打开 `integration_demo/output/demo_report.html` 查看整体效果。该页为单文件 HTML，可直接在浏览器打开，不需要服务器。

## 2. 快速验收

```bash
python -m scripts.verify_project --output-dir artifacts/acceptance_quick
python -m integration_demo.render_report \
  --input-dir artifacts/acceptance_quick \
  --output integration_demo/output/demo_report_quick.html
```

快速验收使用真实 BM25 和 20 条均衡验证样本，覆盖四个领域，以及四个完整 Demo 案例。无需 Dense/Reranker 模型。若目录中已经存在向量文件，去重器会校验并使用它们；移除或更换语料后应重建全部索引，避免使用过期向量。

## 3. 独立单元和图接口测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests integration_demo/tests
python -m compileall -q retrieval scripts integration_demo
```

测试覆盖配置、section 构建及标题 section、语料指纹、BM25 无匹配、RRF 防重复计分、重排依赖注入、去重、请求边界、指标计算、LangGraph 节点及编译子图、Part 1/3 的引用与重试契约。测试不下载模型。

LangGraph 的序列化依赖可能出现一条默认参数将变更的弃用提示；只要测试全部通过，该提示不代表执行失败。

## 4. 从零准备 Python 环境

当前验收环境为 Python 3.12 / macOS arm64。其他平台应使用自己安装的 Python 3.12 创建环境，不要拷贝虚拟环境文件夹。

```bash
python3.12 -m venv .venv_acceptance
source .venv_acceptance/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip --no-cache-dir check
```

`requirements-lock.txt` 记录本机实际验证过的完整版本；`requirements.txt` 记录主要依赖。若包下载慢：

```bash
python -m pip install -r requirements-lock.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 5. 确认数据并重建语料

`multidoc2dial/` 应包含文档 JSON，以及 train / validation / test 三份对话 JSON。本轮仅使用 validation 进行检索验收，不使用 test 调参。

```bash
python -m scripts.build_corpus
```

当前数据预期输出 `Wrote 13421 passages`。领域分布：

| 领域 | passage 数 |
| --- | ---: |
| dmv | 4,313 |
| ssa | 2,460 |
| studentaid | 2,634 |
| va | 4,014 |

每个 passage 对应唯一 `doc_id::id_sec`，包含该 section 的 span ID。标题 section 保留以覆盖标注引用。索引文本包含标题和正文，返回证据中的 `text` 保留 section 文本。数据本身为历史快照，不应当作实时政策服务。

## 6. 下载模型（仅首次或缓存缺失时需要网络）

```bash
unset HF_HUB_OFFLINE
export HF_HOME="$PWD/.model_cache"
python -m scripts.download_models --mirror-first
```

脚本固定本轮模型 revision，对权重做 SHA-256 校验，支持断点续传和主站/镜像切换。省略 `--mirror-first` 优先主站。模型清单保存在 `.model_cache/model_manifest.json`。

使用的正式模型 ID：

- `BAAI/bge-base-en-v1.5`
- `cross-encoder/ms-marco-MiniLM-L6-v2`

第二个 ID 中是 `L6`，不是 `L-6`。

## 7. 重建索引

只构建 BM25：

```bash
python -m scripts.build_indexes --skip-dense
```

构建 Dense（BGE 模型预先下载后可以离线）：

```bash
export HF_HUB_OFFLINE=1
python -m scripts.build_indexes --skip-bm25 --device cpu --batch-size 32
```

Apple 芯片可以把 `--device cpu` 改成 `--device mps`；终端需能访问 GPU。CPU 可用，速度较慢。本轮索引由 MPS 构建，完整检索验收默认使用 CPU，便于重跑。

完整产物应包括：

```text
artifacts/
  passages.jsonl
  bm25.pkl
  bm25_metadata.json
  faiss.index
  passage_embeddings.npy
  dense_metadata.json
```

修改 passage 的正文、标题、顺序或模型后应重建对应索引。指纹校验会拒绝旧索引与新语料混用。只加载自己生成且可信的 `bm25.pkl`。

## 8. 手动逐段验证调用链

单次检索：

```bash
python -m scripts.run_retrieval \
  "How do I renew a driver license?" --domain dmv --mode bm25 --top-k 5
python -m scripts.run_retrieval \
  "How do I renew a driver license?" --domain dmv --mode dense --top-k 5
python -m scripts.run_retrieval \
  "How do I renew a driver license?" --domain dmv --mode rrf --top-k 5
python -m scripts.run_retrieval \
  "How do I renew a driver license?" --domain dmv --mode full --top-k 5
```

预期返回非空 JSON 数组，字段包含 passage ID、正文、span ID、来源和分数；领域均为 `dmv`；passage ID 不重复；每篇文档最多两段。Dense 与 BM25 原始分数不同，不直接相加，RRF 使用名次融合。

独立 Part 1/3 Demo：

```bash
python -m integration_demo.run_demo \
  "How do I renew a driver license?" --mode rrf \
  --output integration_demo/output/single_case.json
```

检查 `plan` 中的领域和子查询、`evidence`、`answer.citations`、`quality.citations_valid`。引用 ID 必须能在证据中查到。当前 Part 3 是抽取式实现，所以不需要额外的生成模型或 API Key。

当前默认入口为 Dense + BM25 + RRF，不加载 Cross-Encoder，也不执行额外语义/每文档去重。RRF 本身仍按 passage ID 合并候选；显式 `full` 模式使用已验证的 current-first user-history query 对 Top 50 重排，并对确认词回答保留 RRF，适合需要更高排序质量且可接受约 1.1–1.2 秒 CPU 平均延迟的场景。

## 9. 看懂验收结果

`artifacts/acceptance/` 保存：

| 文件 | 用途 |
| --- | --- |
| unit_tests.txt | 本次测试输出 |
| evaluation_bm25.json | BM25 逐样本结果 |
| evaluation_dense.json | Dense 逐样本结果 |
| evaluation_rrf.json | 混合融合逐样本结果 |
| evaluation_full.json | RRF + 重排逐样本结果 |
| demo_cases.json | 四领域端到端演示 |
| summary.json | 环境、语料指纹、样本分布、指标和完成状态 |

四种模式使用同一批样本，按领域确定性轮流取样；80 条意味着每领域 20 条。样本 ID 在每份结果里记录。Full 验收禁止重排加载失败后的自动降级。

- Hit@k：前 k 条命中至少一条 gold 的样本比例。
- Recall@k：每条样本覆盖的 gold 比例，再对所有样本平均。
- User Gold Hit Rate / Recall：只在具有 user reference 的样本上，计算 user Gold 的命中率与覆盖率。
- Agent Gold Hit Rate / Recall：只在具有 agent reference 的样本上，计算 agent Gold 的命中率与覆盖率。
- `user_gold_sample_count` 与 `agent_gold_sample_count`：上述两组拆分指标各自的实际样本分母；没有对应 Gold 的样本不按失败计入。
- MRR：首条 gold 的倒数排名均值；当前最大检索截断为 10，可解释为 MRR@10。
- 延迟：检索调用耗时，包含查询编码/融合/重排/去重，不含模型加载。
- 无重复率不代表证据充分，引用结构通过也不代表回答语义完全正确。

总体 Hit@k、Recall@k 和 MRR 使用 User Gold 与 Agent Gold 的去重并集。结果不再输出与它们重复的 `hit_recall@k` 和 `evidence_coverage@k`。

不应把四领域小样本验收结果当成全验证集论文结论。需要全验证集时执行：

```bash
python -m scripts.evaluate_retrieval --mode rrf \
  --output artifacts/evaluation_rrf_validation.json
```

该命令默认全量、4 轮历史、领域过滤；小样本实验可加 `--balanced-by-domain --limit 80`，并用 `--offset 80` 选择下一批留出样本。性能数字会随机器负载变化，验收不硬编码延迟阈值或模型排序优劣。

要判断单路检索是完全找不到 Gold，还是只把 Gold 排在较后位置，运行候选召回诊断：

```bash
python -m scripts.evaluate_candidate_recall \
  --modes bm25 dense rrf \
  --cutoffs 5 10 50 100 200 \
  --limit 80 \
  --output artifacts/candidate_recall.json
```

`bm25` 和 `dense` 展示单路原始候选；`rrf` 展示两路融合后、进入 Cross-Encoder 之前的候选覆盖。为了真实展示 RRF@200，两个分支都会检索到本次最大 cutoff（默认 200），而不是沿用线上 100/50 截断。终端会直接打印 @5/@10/@50/@100/@200 的 Hit Rate、总体 Recall、User Gold Recall 和 Agent Gold Recall；JSON 还保存逐样本 `gold_ranks`，`null` 表示 Top 200 内仍未找到。该入口不经过重排与额外去重。

## 10. 常见问题

- 找不到 Python 包：确认使用 `.venv_acceptance`，再检查 `python -m pip check`。
- PyCharm：将项目解释器选择为当前项目的 `.venv_acceptance/bin/python`；原 `.venv` 不作为本轮验收环境。运行配置的工作目录设为项目根目录，并加入 `HF_HOME` 指向项目的 `.model_cache`。
- 缺少 FAISS 或模型：依次完成下载和索引构建；快速验收可先验证 BM25。
- `RetrievalModelMismatch`：语料/模型/索引不同步，保留旧结果后重建；不要手工绕过指纹。
- 离线找不到模型：确认 `HF_HOME="$PWD/.model_cache"`，然后用下载脚本校验缓存。
- MPS 不可用：切回 `--device cpu`。
- macOS CPU 兼容性：加载器自动使用单线程 Torch 推理，并为 MiniLM 参数创建独立存储，规避本机已复现的底层崩溃与 NaN。验收会拒绝非有限分数。该限制在 macOS CPU 路径生效；会影响进程内 Torch 的 CPU 线程数。
- 原路径消失：本次关闭 iCloud 后原桌面被迁移，项目已复制回本地路径；旧云盘目录和 `.icloud_placeholders_20261027/` 保留用于追溯，正式命令只使用当前源码目录。
- 看到 `runtime_notes`：普通调用允许重排加载失败时降级 RRF；完整验收会直接失败，防止把降级结果误认为 Full 结果。
