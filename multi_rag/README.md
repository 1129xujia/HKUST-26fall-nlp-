# 跨领域 RAG：单轮与评价优化 A/B/C 实验

本目录保留 `main` 模型上的复现代码、固定输入、原单轮结果和新增评价优化结果。请从仓库根目录运行命令。模型与配置取自 `main/configs/integrated.yaml`：MiniLM Embedding、MiniLM Cross-Encoder、DeepSeek Flash；官方 MultiDoc2Dial 语料切成 3798 个 passage。20 题都是 AI 辅助合成的跨领域请求，仍需独立人工复核。

## 文件与接口

| 文件 | 输入 → 输出 |
|---|---|
| `data/cross_domain_20_annotations.json` | 20 条合成问题、领域及必答点标注；在线流程只读取问题文本 |
| `prepare_main.py` | 读取 `main` 固定模型版本和仓库 MultiDoc2Dial 原始数据 → 下载模型、生成 3798 个 passage 及校验索引；输出到本目录的 `artifacts/`、`data/` |
| `run_main_experiment.py` | 问题 → `plan()` 返回 `QueryPlan` 和预测领域；`retrieve_branch()` 调用 main 的 `Part2RetrievalAdapter`；`answer_main()` 调用 main 的 `EvidenceResponseGenerator`；C 的 `summarize_main()` 最后汇总子答案和原始证据 → `results/<UTC批次>/cases.jsonl` |
| `run_full_pipeline.py` | 读取已保存的单轮答案和证据 → main 的 `MainEvaluatorAdapter`、`OptimizationController`；按 A/B/C 约束继续检索、修订和评价 → 新批次 `cases.jsonl` |
| `evaluate_main_answers.py` | 批次答案、证据与固定标注 → 逐题必答点评分、来源文档检索指标、引用与成本汇总 → `point_grades.jsonl`、`main_summary.json`、`case_summary.csv` |
| `results/20261107T194151Z/` | 原单轮 20 题结果；保持原样 |
| `results/20261108_full/` | 基于原答案续跑 main 评价与优化后的 20 题结果，含反馈、动作及停止原因 |

三组共用一次多标签规划：`plan()` 只把问题文本交给模型预测领域；标注领域仅用于事后核对和评分。A 组 `domain=None` 全库检索，Dense/BM25 各 Top-50、RRF 候选 30、重排后取 6 条证据并统一生成。B 组对每个预测领域独立过滤检索（各检索器 Top-25、RRF 候选 15），各取 3 条证据，合并后统一生成。C 组在各领域**独立完成检索、重排和子答案生成**，最后才汇总子答案及原始证据。原单轮批次的 B/C 证据及顺序逐题相同。

新增批次复用原单轮答案作为首轮候选，避免重复付费生成；每个候选都进入 main 的评价器和优化控制器，按 main 默认阈值、最多 2 次重试、30,000 Token 和 180 秒的分支预算运行到停止。A 优化时仍查全库；B 优化时仍对预测领域分别过滤检索；C 每个领域分别评价和优化，领域证据不交叉，然后重新汇总并评价、修订最终答案。C 的最终修订只可复用已汇总的证据，不能再检索跨领域新证据。最终上下文最多 6 条。单领域请求不在本次对照中。

## 启动流程

1. 从仓库根目录选定已安装 PyTorch、Transformers、safetensors、NumPy、PyYAML、Jinja2 的 Python 3.11 环境；本次运行使用 Python 3.11.18、Torch 2.9.1、Transformers 4.57.1。仓库须已有 `multidoc2dial/data/multidoc2dial/` 原始数据。本机可这样指定 Python；其他机器替换该路径。将 main 代码所需的其余依赖装在本目录：

   ```powershell
   $ragPython = 'D:/Users/Astrella/anaconda3/envs/multidoc-rag/python.exe'
   & $ragPython -m pip install --target main/multi_rag/vendor pydantic==2.11.7 rank-bm25==0.2.2 python-dotenv==1.1.1
   ```

2. 准备模型和索引（需要访问公开模型下载源；模型 revision 来自 `main`）：

   ```powershell
   & $ragPython main/multi_rag/prepare_main.py
   ```

3. 在当前进程设置 `DEEPSEEK_API_KEY`，运行 20 题。密钥不要写入代码、命令参数或结果文件：

   ```powershell
   $secureKey = Read-Host 'DeepSeek API key' -AsSecureString
   $env:DEEPSEEK_API_KEY = [System.Net.NetworkCredential]::new('', $secureKey).Password
   & $ragPython main/multi_rag/run_main_experiment.py
   ```

   可用 `--limit 1` 做单题检查；中断后用 `--resume-dir` 加上输出的批次目录从已保存的题目继续。每条答案完成就写入 `cases.jsonl`。

4. 对单轮结果做离线必答点评分（同样需要密钥）；若复用目录中保留的原批次，直接使用该批次：

   ```powershell
   $runDir = 'main/multi_rag/results/20261107T194151Z' # 换成新运行输出的目录
   & $ragPython main/multi_rag/evaluate_main_answers.py --run-dir $runDir
   ```

5. 从保存的单轮答案继续运行 main 评价、优化和 C 的新汇总；中断后用相同 `--run-dir` 续跑，已完成的题目会跳过：

   ```powershell
   $fullRun = 'main/multi_rag/results/new_full_run' # 也可省略 --run-dir 自动建批次
   & $ragPython main/multi_rag/run_full_pipeline.py --source-dir $runDir --run-dir $fullRun
   ```

6. 对新批次做离线必答点评分并汇总指标：

   ```powershell
   & $ragPython main/multi_rag/evaluate_main_answers.py --run-dir $fullRun
   Remove-Item Env:DEEPSEEK_API_KEY
   ```

原批次和新批次各有独立 `manifest.json`。新批次校验原批次文件及索引语料的 SHA256。模型、索引、预处理缓存和安装依赖可由第 1–2 步重建，最终结果目录不保留它们。评分脚本在 20 条 `point_grades.jsonl` 已存在时只重新计算汇总，不再调用模型。

## 原单轮结果：`20261107T194151Z`

20 题均为 AI 辅助合成的跨领域请求。运行时工作树为 `ccf5933`；远程 `main` 的 `50d7acb` 只新增 validation A 报告及脚本，本实验调用的配置和运行模块没有变化。官方语料按 main 流程切为 **3798 个 passage**；模型为 `all-MiniLM-L6-v2@1110a243`、`ms-marco-MiniLM-L6-v2@233902d2` 和 `deepseek-flash`。20 题的预测领域集合均与标注相同，C 分支证据隔离和全部引用 ID 均通过检查。

| 指标 | A 全库统一生成 | B 分领域检索、统一生成 | C 独立领域 RAG、最终汇总 |
|---|---:|---:|---:|
| 指定来源文档召回率 | 60.0% | 75.0% | 75.0% |
| 来源文档检索 F1@6 | 40.7% | 48.4% | 48.4% |
| 必答点覆盖率（同模型初评） | 48.1% | 56.8% | 60.7% |
| 全部必答点覆盖的题数 | 1/20 | 2/20 | 3/20 |
| 平均生成调用次数/题 | 1.0 | 1.0 | 3.1 |
| 平均生成 Token/题 | 1732 | 1759 | 4528 |
| 平均生成耗时/题 | 1.92 秒 | 2.14 秒 | 5.67 秒 |

B 相对 A 的指定来源文档召回增加 15 个百分点（逐题 7 胜、12 平、1 负）；必答点覆盖增加 8.7 个百分点（7 胜、11 平、2 负，成对 bootstrap 95% 区间约 −0.4 至 +17.9 个百分点）。C 在相同证据下相对 B 多覆盖 3.9 个百分点，只有 014、017、018 三题增加必答点，17 题持平；平均生成 Token 是 B 的 2.58 倍。B 的检索覆盖值得继续验证，C 的小幅回答收益尚不足以证明额外成本划算。

标注仍待独立人工复核；指定来源文档召回未归并全库等价文档；必答点评分由与生成同系列模型完成。20 题多直接写明机构名称，20/20 路由一致率不能代表未见问法的泛化能力。领域配额还使 002、009 两题的 B 答案比 A 漏掉更多必答点。逐题记录见 [`results/20261107T194151Z/case_summary.csv`](results/20261107T194151Z/case_summary.csv)。

## 评价优化后结果：`20261108_full`

以下是同 20 题、同首轮答案和证据经 main 评价器与优化器运行到停止后的结果；它与上表的单轮结果分别保留。C 的领域子流程成功通过评价 22/40，另有 17 个升级处理、1 个生成校验错误；最终汇总通过评价 12/20。

| 检索及最终回答指标 | A 全库 | B 分领域检索 | C 独立领域 RAG |
|---|---:|---:|---:|
| 来源文档 Recall@1 / @3 / @6 | 27.5% / 42.5% / 55.0% | 20.0% / 52.5% / 75.0% | 20.0% / 50.0% / 67.5% |
| 来源文档 Precision@6 | 30.4% | 35.4% | 31.2% |
| 来源文档 F1@6 | 37.1% | 47.8% | 42.5% |
| 至少命中一份指定文档 / 全部指定文档命中 @6 | 85% / 25% | 95% / 55% | 90% / 45% |
| 标注 span 召回@6 | 36.7% | 42.1% | 40.3% |
| 必答点覆盖率（同模型初评） | 12.5% | 45.0% | 43.2% |
| 全部必答点覆盖的题数 | 1/20 | 2/20 | 3/20 |
| 最终 `answered` / `escalated` | 3 / 17 | 14 / 6 | 12 / 8 |
| 引用证据 ID 有效率（有引用的答案） | 100% | 100% | 100% |
| 引用落在指定来源文档的比例 | 55.6% | 68.5% | 58.8% |

| main 评价、优化与成本 | A 全库 | B 分领域检索 | C 独立领域 RAG |
|---|---:|---:|---:|
| 首轮评价通过 → 优化后通过 | 2 → 3 /20 | 13 → 14 /20 | 11 → 12 /20 |
| 末次候选：相关性 / 充分性 | 70.5% / 56.0% | 91.8% / 79.8% | 90.0% / 72.8% |
| 末次候选：有据性 / 对话一致性 | 91.0% / 99.0% | 90.8% / 98.3% | 92.0% / 97.3% |
| 平均评价次数 / 题 | 1.70 | 1.60 | 4.35 |
| 平均优化动作 / 题 | 1.40 | 0.65 | 2.40 |
| 平均生成调用次数 / 题 | 1.75 | 1.60 | 4.70 |
| 平均流程模型调用次数 / 题 | 3.45 | 3.20 | 9.05 |
| 平均生成 Token / 题 | 3532 | 3298 | 7179 |
| 平均流程 Token / 题 | 7313 | 6866 | 14756 |
| 从原答案继续新增耗时 / 题 | 7.68 秒 | 5.15 秒 | 13.78 秒 |

完整流程的 B 相对 A：必答点覆盖高 32.5 个百分点（逐题 10 胜、10 平、0 负；成对 bootstrap 95% 区间约 +17 至 +48 个百分点），最终可回答题多 11 题。C 相对 B：必答点覆盖低 1.7 个百分点（1 胜、18 平、1 负；区间约 −9 至 +3.8 个百分点），平均流程 Token 为 B 的 2.15 倍。C 在 009、014、018 题优化后更换了领域证据，指定来源文档召回因此降至 67.5%；009 的 SSA 子流程还发生一次 main 生成结果校验错误。优化器使 A/B/C 各多 1 题通过评价，但对未通过的答案多数选择升级处理，故最终必答点覆盖低于原单轮表。逐题结果见 [`results/20261108_full/case_summary.csv`](results/20261108_full/case_summary.csv)，完整反馈和动作在同批次 `cases.jsonl`。

## 指标口径与限制

- **检索 Precision/Recall/F1@k**：在最终送入生成器的前 k 条证据 passage 中，以标注的 `(领域, 来源文档)` 为真值，先按来源文档去重。Precision 是命中的指定文档数/检出文档数；Recall 是命中的指定文档数/指定文档数；F1 是每题 Precision 与 Recall 的调和平均。表中均为 20 题的逐题宏平均。`@1/@3/@6` 是**最终上下文**指标；main 原有的 `@1/@5/@10` 计算的是重排后的候选列表，阶段不同，不直接对比。指定文档以外的等价资料尚未归并，Precision/F1 只能作本标注集内的诊断。
- **命中率/完整覆盖率、span 召回**：命中率是至少找到一份指定文档的题目比例；完整覆盖率是找到全部指定文档的题目比例；span 召回是最终 passage 的 `span_ids` 覆盖标注 `span_id` 的比例，先按题计算再平均。
- **必答点覆盖率**：逐题按固定标注的必答点计算，非 `answered` 状态计 0，再对 20 题平均。由与生成同系列的 DeepSeek Flash 初评，不等于人工事实核验。main 的四项评价分数也是模型对**末次被评价候选**的打分，不能当作独立真值；`evaluation_passed` 表示满足 main 阈值并由优化控制器放行。
- **引用与成本**：引用有效率检验所引 passage ID 是否属于最终证据，非回答事实正确率；引用指定来源比例按引用 passage 所在文档计算。流程 Token/调用数包含复用的首轮生成、main 在线评价和修订；C 还包含两个领域子流程和新最终汇总。三组共用的规划平均 321 Token/题单列，不计入各组；额外的离线必答点评分共用 24,573 Token，也不计入各组。新增耗时只统计从保存的首轮答案继续的步骤，不含原规划、首轮检索和首轮生成。
- **未给出的 main 指标**：本标注集没有独立标准答案，无法计算 main 的回答 Token-F1 与 Exact Match；也没有逐声明的人工支持标注，不能给出可靠的幻觉率、事实有据率或任务完成率。`main_summary.json` 将回答 F1/EM 记为 `null`，不能把上表的**检索文档 F1**当成回答 F1。20 题多直接点明机构，领域路由 20/20 只说明这组题目；全部标注尚待独立人工复核。
