# 实验与结果

实验代码统一在 `experiments/`，通过仓库根目录的 `python -m experiments.<模块>` 运行；共享的检索、生成和优化实现分别来自统一源码包中的 `customer_service_rag.retrieval`、`customer_service_rag.generation` 和 `customer_service_rag.optimization`。无需将仓库重命名为 main，也不再安装 vendor 副本。使用根项目环境运行客服实验；BGE训练和旧检索研究使用兼容的独立推理环境。

## 数据与评价协议

- MultiDoc2Dial 是当前实现的数据集，包含 dmv、ssa、studentaid、va 四领域。共享字符窗口语料3798块；BGE另支持 section 和 struct（完整 struct 为4110块）。不同切分须使用对应的索引和 Gold。
- 官方 v1.0 数据包明确说明 test 文件是 **dummy**。历史微调 test400 的冻结哈希已确认来自这个文件，因此这些分数仅作开发对照，不能作为真实官方测试集成绩。重新准备该历史划分须显式传入 `--development-only-test`；准备和评价不会默认把它当正式测试集。
- 官方 validation 已被团队用于检索调参和1000题开发运行，不能把全量 validation 描述为从未接触的最终测试集。正式结果需要事先冻结、与训练和调参隔离的有效评价样本，并记录来源、ID、随机种子和模型/索引身份。
- 20题跨领域请求为AI辅助合成，必答点标注尚待独立人工复核。相同系列模型评分和在线评价通过均不等于事实正确。
- 引用ID有效率只说明引用属于给定证据。无独立人工标注时，幻觉率、事实有据率和任务完成率保持 null。HotpotQA 尚未适配。

## 官方 validation 单次流程

代码：`experiments/validation/run.py`、`finalize.py`。运行器复用 `configs/deepseek.yaml`，明确关闭 quality、optimization 和 escalation；其余规划和混合检索/重排与主流程一致，因此它不是 Dense-only Single-RAG。

```powershell
python -m experiments.validation.run --count 1000 --seed 42
python -m experiments.validation.finalize --run-dir results/validation_a_seed42_n1000
```

真实运行需要本地密钥、官方预处理数据和已建好的共享索引。`--config` 可指定其他真实模型配置。每次尝试立即保存轨迹和进度；error 可触发外层重试。新汇总分别提供 `last_attempt` 和 `all_attempts`，后者累计全部已保存尝试的已知 Token、调用数和请求耗时，并单列未知用量；耗时不含重试退避、模型加载及建索引。`--output-dir` 可将重算报告写到新目录。

固定历史交付位于 `reports/validation_a_seed42_n1000/`：1000条中973 answered、25 insufficient_evidence、2 error；Token F1约0.2080、EM=0、Evidence Recall@5约0.6813。selection、config、predictions、summary和metrics保持原样，旧成本字段只统计每题最后一次尝试，不是包含重试的总成本。复算全部尝试成本需要本地 `results/` 的原始轨迹。

## 跨领域 A/B/C

唯一标注：`data/cross_domain/cross_domain_20_annotations.json`。

| 组别 | 行为 |
|---|---|
| A | 全库混合检索、统一生成 |
| B | 按预测领域分别检索，再统一生成 |
| C | 各领域独立检索和生成，再使用原始证据汇总 |

每题三组共享同一次领域规划，Gold只供事后评分。初轮总上下文最多6条，B/C使用相同证据配额。续跑时A仍全库检索、B仍分领域检索、C的领域流程隔离；C最终修订只能复用已汇总证据。各分支预算和成本分别记录。

```powershell
# 复用根目录模型/数据/索引；仅显式开关才发起资产下载
python -m experiments.cross_domain.prepare
python -m experiments.cross_domain.run --limit 20
python -m experiments.cross_domain.optimize --source-dir reports/cross_domain/20261107T194151Z
python -m experiments.cross_domain.evaluate --run-dir results/cross_domain/<新批次>
```

`prepare --download-data --download-models` 调用共享资产下载函数，不维护第二份模型下载器或索引。run、optimize及缺少评分记录时的evaluate会调用真实LLM；开发重构检查不需要执行它们。run通过 `--resume-dir` 续跑；optimize使用相同 `--run-dir` 可续跑。评价失败的草稿只留作诊断，最终状态标为error，不会当作通过质量控制的答案计分。

历史两批完整记录位于 `reports/cross_domain/20261107T194151Z/` 和 `reports/cross_domain/20261108_full/`，保留cases、逐点评分和manifest的原始内容，其中旧路径仅表示当时的运行环境。

| 指标 | A | B | C |
|---|---:|---:|---:|
| 单轮必答点覆盖（同模型评分） | 48.08% | 56.75% | 60.67% |
| 评价优化后必答点覆盖 | 12.50% | 45.00% | 43.25% |
| 评价优化后 answered | 3/20 | 14/20 | 12/20 |

非answered计零，因此转接会降低最终必答点覆盖。这些小规模合成实验不能证明泛化优势；来源文档F1也不能替代回答Token F1。

## BGE微调与检索消融

`experiments/finetune/` 保留队友的数据准备、多正例对比训练、dev选模、A–D评价与对话聚类bootstrap分析，直接复用正式 `customer_service_rag.retrieval` 模块，不再保存一份runtime代码。独立训练依赖见 `experiments/finetune/requirements-autodl.txt`；保留已有CUDA PyTorch，不将旧版NumPy/Transformers强装进主项目环境。

```bash
# 仅用于重建历史dummy开发队列，不是正式测试准备
python -m experiments.finetune.prepare --development-only-test

# 默认本地模型路径可用 BASE_MODEL、RERANKER_MODEL 覆盖
bash experiments/finetune/run.sh train
bash experiments/finetune/run.sh select
DEVELOPMENT_ONLY_TEST=1 bash experiments/finetune/run.sh test
```

默认数据从共享官方目录查找；生成语料位于 `artifacts/indexes/finetune/`，训练输入、checkpoint、日志和评价产物位于实验目录的Git忽略子目录。train默认CUDA全参数2000步、batch8；只用dev选checkpoint，已有结果拒绝覆盖。CPU开发可直接调用train CLI并显式设置较小步数，不能将小规模运行当成正式训练。

已提交权重及原始实验信息位于 `models/bge-base-en-v1.5-struct-ft/`，包括 Git LFS权重、清单、选模和对照记录。历史dummy开发队列中A原始Dense为59.75%、B微调Dense为78.25%、C混合召回为66.25%、D混合加重排为74.50% Hit@10；B与D差值置信区间跨零，且受dummy数据来源约束，不能据此宣布稳健优越性。此模型接入时必须重编码文档，不能使用原始BGE或MiniLM的向量索引。

`experiments/retrieval/` 只保留统一profile评价和真实资产验收两个入口。索引构建统一使用 `cs-rag index`，不再保留手写索引元数据的旧分块构建器或独立探索脚本。

```bash
python -m experiments.retrieval.evaluate --help
python -m experiments.retrieval.validate --help
```

evaluate读取已冻结的query/Gold样本和对应完整索引，计算Hit、Recall、MRR、nDCG及延迟，拒绝Gold不在索引中的样本；切分和编码器必须一致。validate按section/struct与rerank/finetuned四种组合验证真实检索和引用，生成/评价使用mock。两者保留模型/语料指纹与索引错配检查，真实编码器仍需对应本地模型。用小样本文档建立的验收索引不能用于完整数据集指标评价。

真实编码器、mock生成/评价的小样本接入记录保留在 `reports/retrieval_profiles/local_validation.json`，用于验证接口，不是质量基准；新报告通过 `--output results/...` 写入运行目录。
