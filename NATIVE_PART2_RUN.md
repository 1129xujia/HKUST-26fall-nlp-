# 本地 BGE/FAISS 接入最新 main

结论：可以跑通。已在 main e9dc3d0 的副本中新增 Part 2 原生入口，复用 main 现有 Planner、EvidenceResponseGenerator、MainEvaluatorAdapter、OptimizationController 和共享 BudgetLedger。

新增文件：

- `src/retrieval_v2/native_main.py`：原生检索连接与 CLI。
- `tests/test_native_part2_main.py`：3 项跨模块边界测试。

原有业务文件、main 的 `cs-rag` 默认行为和本地索引均未改动。该入口不需要官方字符窗口预处理数据，但需要已有 Part 2 section 索引和 BGE 模型。

## 已验证

| 模式 | 检索 | 生成/评价 | 结果 |
| --- | --- | --- | --- |
| RRF | 本地 BGE + FAISS + BM25 + RRF | 最新 main 的 mock | answered / evaluation_passed |
| Full | 上述链路 + 真实 MiniLM Cross-Encoder | 最新 main 的 mock | answered / evaluation_passed |

同一问题 `How do I replace a lost driver's license?`：RRF 首条为 `Get a driver license#1_0::11`，Full 首条为 `Replace license or permit#1_0::40`，真实重排分数 6.9351606369。两条路径各执行一次召回、一次评价，保持原始 doc::section 引用。

3 项新增测试通过：原生 ID 在生成/评价后保留；main 控制器仅重新生成时不重复召回；初次预算不重置。它们使用固定组件，不是正式质量评价。

输出：`results/native_bge_rrf.json` 和 `results/native_bge_full.json`。

## 当前电脑运行

复用原来已验证的模型虚拟环境；`team_integration/.deps` 只补充其中缺少的 python-dotenv。最新 main 的独立 `.venv` 仅安装了基础 CLI 依赖，没有 BGE/FAISS 推理依赖，因此此命令显式使用外层环境。

```bash
cd /Users/leo/Desktop/hkust/5018/project/main_validation_20261106_wVLjJi
export HF_HOME="/Users/leo/Desktop/hkust/5018/project/.model_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PYTHONPATH=src:../team_integration/.deps ../.venv/bin/python \
  -m retrieval_v2.native_main \
  "How do I replace a lost driver's license?" \
  --artifact-dir ../artifacts \
  --mode rrf --llm mock --top-k 3 --domain-filter \
  --output results/native_bge_rrf.json
```

`--mode full` 开启真实重排；不传 `--domain-filter` 时全局搜索。`--llm deepseek` 改用 main 的真实客户端，需要环境变量 DEEPSEEK_API_KEY；本次没有发送真实 API 请求。

## 边界

这个入口沿用 main 的处理组件，但输出原生 Evidence 与 OptimizationResult 的 JSON，没有强行转换成不接受 `::` 的旧 main Passage，也未接入旧字符窗口的 RunResult/Gold 评价。已有 `cs-rag` 离线评价不能直接拿字符窗口 Gold 去评价这些 section ID。

此次检索默认值为 Dense100 / BM2550 / RRF30 / candidates50；使用独立 rewritten_query，子问题分解关闭。它与之前 history + gold domain 的 validation 设置不同，不宣称复现了那批 Recall/Hit 指标。

已确认真实检索和重排可接入最新组件；真实 DeepSeek 生成/评价、正式 validation 和新环境完整安装依赖仍待单独验证。未创建 commit 或向 GitHub 推送。
