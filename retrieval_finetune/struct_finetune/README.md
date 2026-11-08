# Struct Dense 微调实验

本目录保存已完成的 BGE struct 微调实验代码：数据构造 → 训练 → 验证选模 → 四组最终测试 → 配对分析。独立于应用服务，不更改默认检索模型。

## 文件入口

| 文件 | 职责 |
|---|---|
| `prepare.py` | 生成 struct 语料；按原始对话划分构造样本、审计重叠、合并同一输入的多正例 |
| `train.py` | 多正例对比学习、保存 checkpoint、重新加载并检查数值一致性 |
| `evaluate.py` | 为指定模型重建语料向量；输出 Dense、RRF 和可选重排的逐 query 排名与指标 |
| `analyze.py` | 仅用 dev 选 checkpoint；比较 A–D，生成对话聚类 bootstrap 区间和差异样本 |
| `common.py` | 查询契约、抽样、指标及 JSON/SHA256 工具 |
| `run.sh` | prepare、smoke、train、select、test、export 分阶段入口 |
| `runtime/retrieval/` | 原实验使用的检索模块快照；与仓库现有服务隔离，保留 struct 和评分语义 |
| `test_experiment.py` | 多正例指标、未来回复隔离、损失及梯度检查 |

代码及配置纳入 Git；数据、模型、向量索引、日志和运行结果不随本目录提交。这里没有打包微调权重，也不需要 Git LFS。

## 环境与数据准备

以下命令均在仓库根目录执行。建议独立训练环境，避免本实验固定依赖与主应用环境的版本冲突。AutoDL 直接使用镜像自带的 CUDA PyTorch，不重新安装 PyTorch：

```bash
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
python -m pip freeze | python -c 'import sys; print("".join(s for s in sys.stdin if s.lower().startswith(("torch==", "torchvision==", "torchaudio=="))), end="")' > /tmp/struct-torch-constraints.txt
python -m pip install -r retrieval_finetune/struct_finetune/requirements-autodl.txt -c /tmp/struct-torch-constraints.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

原实验采用 PyTorch 2.6.0；其他镜像版本须先跑 smoke。约束文件会阻止 pip 替换现有 Torch 版本；若依赖冲突，使用兼容镜像，不删除约束强行覆盖。

将原始文件放到 `multidoc2dial/`，或通过 `RAW_DIR` 指定包含下列文件的目录：

- `multidoc2dial_doc.json`
- `multidoc2dial_dial_train.json`
- `multidoc2dial_dial_validation.json`
- `multidoc2dial_dial_test.json`

```bash
RAW_DIR=/path/to/multidoc2dial bash retrieval_finetune/struct_finetune/run.sh prepare
```

生成 `artifacts_struct_bge/passages.jsonl` 和本目录 `data/{train,dev,test,audit}.json`。默认全量去重训练样本、dev160、test400，按领域平衡抽样。已有语料会先校验与原始文件生成的 struct 一致性，防止静默覆盖错误语料。

已上传的 AutoDL 包中 `artifacts_struct_bge/` 与 `retrieval_finetune/struct_finetune/data/` 也可直接复制到仓库同名位置，保留冻结样本，不必重新 prepare。

## 模型与运行步骤

默认离线加载 `local_models/bge-base-en-v1.5` 和 `local_models/ms-marco-MiniLM-L6-v2`。可以复制原 AutoDL 包内的这两个目录；或下载后通过 `BASE_MODEL`、`RERANKER_MODEL` 指定本地目录。需要在线下载时先显式设置 `HF_HUB_OFFLINE=0` 运行下载命令，运行脚本本身使用离线模式。

```bash
python -m pytest -q retrieval_finetune/struct_finetune/test_experiment.py
bash retrieval_finetune/struct_finetune/run.sh smoke
bash retrieval_finetune/struct_finetune/run.sh train
bash retrieval_finetune/struct_finetune/run.sh select
bash retrieval_finetune/struct_finetune/run.sh test
bash retrieval_finetune/struct_finetune/run.sh export
```

`smoke` 用 2 步训练验证保存/重载，输出到独立 `smoke_gpu/`，不是效果实验。`train` 默认 CUDA 全参数训练 2000 步、batch8、每1000步保存；已有训练输出时拒绝覆盖。`select` 只按 dev 的 Hit@10、MRR@10、nDCG@10 顺序选模。`test` 冻结最终结果后拒绝再次覆盖；`export` 生成 `autodl_results.tar.gz`，包含选定权重、索引、配置及结果，不应直接加入普通 Git。

可以用 `PYTHON`、`DEVICE`、`STEPS`、`BATCH_SIZE`、`CHECKPOINT_EVERY` 环境变量覆盖运行参数。CPU smoke 可用 `DEVICE=cpu`，正式 train 入口要求 CUDA。

## 训练与评测契约

- 基础模型 BAAI/bge-base-en-v1.5；768维 CLS pooling，最大512 tokens，归一化向量。
- query 前缀：`Represent this sentence for searching relevant passages: `；passage 无前缀，使用标题路径和正文组成的 `index_text`。
- query 包含最近4个可见 turn 和角色标签；下一条 agent 回复仅用于离线 gold，不进入输入。
- AdamW，lr2e-5，weight decay0.01，warmup10%后衰减，梯度裁剪1.0，seed20261108。
- 多正例 logsumexp 对比损失，scale20；合并同输入已知正例，mask 同文档未标注候选，不将其直接当负例。
- A：原始 Dense；B：微调 Dense 无重排；C：原始 Dense100 + BM2550 + RRF(k30)，保留50；D：同一批50候选经 Cross-Encoder 重排。评估 Hit/Recall/MRR/nDCG@10、覆盖率、领域指标与延迟。
- 模型权重更新后必须重新编码文档；`evaluate` 根据权重和语料身份校验缓存。不要将微调后的查询向量用于原始模型的索引。

原始实测选中 checkpoint-2000；test400 的 Hit@10：A59.75%、B78.25%、C66.25%、D74.50%。这些是历史实验参考，不是本次迁移重新训练的结果；运行结果以保存的 JSON 为准。

## 验收

macOS 如果出现 `objc ... initialize ... fork()`、`Abort trap: 6`，且崩溃进程为 `/opt/local/bin/bash`，是 MacPorts bash/libintl 在启动 Python 前的进程问题。脚本已在 macOS 自动用 `/bin/bash` 重新执行；也可以直接使用 `DEVICE=cpu /bin/bash retrieval_finetune/struct_finetune/run.sh smoke`。不要通过关闭 Objective-C fork 安全检查来掩盖错误。

`pytest` 应通过3项测试。smoke 应生成 `smoke_gpu/checkpoint-2/` 和 `training_summary.json`，`reload_max_abs_error` 不超过1e-4。select 输出 `run/selection.json`；test 输出 `results/final_comparison.json/.md` 和 `discordant_cases.json`，每条排名保存在 base/tuned test JSON。小语料调试时至少需要200个 passage（评测固定搜索200候选），正式实验应使用完整语料。
