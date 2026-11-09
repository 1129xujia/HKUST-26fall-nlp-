# 在线客服多智能体 RAG

HKUST 26fall NLP 项目。以队友实现的并行检索、带引用生成和 Evaluator–Optimizer 为正式模块，通过一个 `cs-rag` 入口运行。

## 项目报告

[8页 ACL 英文报告](reports/project_report/customer_service_rag_report.pdf)、[中文说明与待完成工作](reports/project_report/customer_service_rag_report_notes_zh.docx)、[可编辑 LaTeX 源码](reports/project_report/latex/) 和队友的检索原稿统一放在 [reports/project_report](reports/project_report/README.md)。总报告已区分当前微调 Dense 直排配置与历史 MiniLM 混合重排实验。

## 目录

```text
configs/                 # default、deepseek、native 和实验配置
src/
  customer_service_rag/  # 统一源码包，提供 cs-rag 和 Python 模块入口
    data/                # 数据处理、样例、grounding 和共享索引
    planning/            # 上下文理解、查询改写和领域规划
    retrieval/           # 队友的 Dense、BM25、RRF、重排、section/struct 切分
    generation/          # 证据限定生成与引用校验
    optimization/        # 队友的评价适配、优化控制器、路由、预算和召回缓存
    models/              # 模型适配
    orchestration/       # 统一编排和检索资产适配
    evaluation/          # 离线评价
    observability/       # 指纹、轨迹和结果记录
experiments/             # cross_domain、finetune、retrieval、validation 研究入口
reports/                 # 固定实验结果和标注审阅记录
models/                  # Git LFS 微调权重与原始实验来源记录
data/                    # 共享样例、唯一跨领域标注；原始/预处理数据不入 Git
tests/                   # 按模块整理的既有测试
```

架构和接口见 [docs/architecture.md](docs/architecture.md)，数据协议、实验命令和已知结果见 [docs/experiments.md](docs/experiments.md)。

## 离线运行

`cs-rag` 和 `python -m customer_service_rag` 调用同一 CLI。以下使用 Python 模块入口。在仓库根目录执行，Python 3.10+。已存在 `.venv` 时可直接使用；包目录重构后需要重新进行可编辑安装。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m customer_service_rag doctor
.\.venv\Scripts\python.exe -m customer_service_rag index
.\.venv\Scripts\python.exe -m customer_service_rag ask --message "Which form replaces a lost driving license?"
.\.venv\Scripts\python.exe -m customer_service_rag run --limit 2
```

默认配置使用16条虚构客服样例、HashEmbedding 和 MockLLM，并执行正式模块的评价/优化控制流程。它无需密钥、网络或GPU；mock分数只用于开发，不代表语义质量。`--no-save` 关闭运行记录，`--full` 展开 ask/retrieve 的完整结果。

## 真实 MultiDoc2Dial 与 DeepSeek

`configs/deepseek.yaml` 使用官方语料、已有的384维 MiniLM Embedding、MiniLM Cross-Encoder 和 DeepSeek Flash。它继续使用现有共享 NumPy 索引，检索融合、重排和优化执行均调用队友模块。

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[local]"
# 在被 Git 忽略的 .env 中设置 DEEPSEEK_API_KEY
.\.venv\Scripts\python.exe -m customer_service_rag download-data --config configs/deepseek.yaml
.\.venv\Scripts\python.exe -m customer_service_rag download-models --config configs/deepseek.yaml
.\.venv\Scripts\python.exe -m customer_service_rag preprocess --config configs/deepseek.yaml
.\.venv\Scripts\python.exe -m customer_service_rag index --config configs/deepseek.yaml
.\.venv\Scripts\python.exe -m customer_service_rag ask --config configs/deepseek.yaml --message "How can I replace a lost driver's license?"
```

当前工作区已有共享官方数据、MiniLM模型和索引，运行前可用 doctor 查看资源。下载命令默认只取 Embedding 和 Reranker，`--include-llm` 额外下载可选本地LLM。API密钥只放本地环境变量或 `.env`。

## 原生 BGE / FAISS

`configs/native.yaml` 选择队友的 struct + 微调Dense 链路，与上述配置使用相同规划、生成、评价、优化和结果保存。该链路直接保留 Dense 排名，不启用 BM25/RRF/重排。原始 BGE + BM25 + RRF + Cross-Encoder 可通过 `retrieval.chain: rerank`、对应分支和模型配置选择；section/struct、原始/微调模型各使用独立索引目录。

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[local,native]"
git lfs pull --include="models/bge-base-en-v1.5-struct-ft/checkpoint-2000/model.safetensors"
.\.venv\Scripts\python.exe -m customer_service_rag doctor --config configs/native.yaml
.\.venv\Scripts\python.exe -m customer_service_rag index --config configs/native.yaml
.\.venv\Scripts\python.exe -m customer_service_rag ask --config configs/native.yaml --message "How do I renew my license?"
```

微调模型位于 `models/bge-base-en-v1.5-struct-ft/checkpoint-2000`。原生后端需要额外依赖和新建 FAISS 索引，不能使用现有 MiniLM 向量。索引构建会核对模型/语料身份；原生索引目录存在时拒绝覆盖，更新模型时应设置新的 `paths.index_dir`。

## 常用命令

```powershell
# 只检索，不请求 LLM
.\.venv\Scripts\python.exe -m customer_service_rag retrieve --query "license replacement" --full
# 多轮输入：history.json 为 [{"role":"user","text":"..."},{"role":"agent","text":"..."}]
.\.venv\Scripts\python.exe -m customer_service_rag ask --message "What about next year?" --history history.json
# 相同真实模型和混合检索，只关闭评价、优化与转接
.\.venv\Scripts\python.exe -m customer_service_rag run --experiment single --split mini --limit 4
# Dense-only 基线；另有 bm25、hybrid、full 配置
.\.venv\Scripts\python.exe -m customer_service_rag run --experiment dense --split mini --limit 4
# 对实际生成的批次评价，替换目录名
.\.venv\Scripts\python.exe -m customer_service_rag evaluate --run-dir results/<batch_id>
```

单次运行只保存 `config.yaml`、`metadata.json`、`result.json`，完整模型调用、候选、引用、反馈、动作和轨迹都在 result 内；批次另存 predictions.jsonl。保留既有结果字段，方便读取历史实验。运行产物、下载模型、原始数据和索引不入Git；已冻结的研究结果保存在 reports。

## 当前边界

已接通 MultiDoc2Dial 的完整主流程。官方 v1.0 test 文件为 dummy，历史微调 test400 仅作开发参考；validation也已用于开发，正式最终评价需要重新冻结有效且隔离的样本。20题跨领域实验是待独立人工复核的合成集。HotpotQA尚未适配，人工转接只生成结构化包。当前按原约定暂缓执行测试套件；保留并整理已有测试。
