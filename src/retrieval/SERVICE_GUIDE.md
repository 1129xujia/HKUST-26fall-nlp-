# 主服务：切分与检索链路选择

新增代码位于 `src/retrieval/`，不依赖微调实验目录。原生入口 `retrieval.native_main` 接通现有 Planner → 检索 → 生成 → 评价/优化控制器。旧的 `cs-rag` 字符窗口索引入口仍独立，不会因放入 checkpoint 自动切换。

## 四种组合

| segmentation | chain | 行为 | 默认索引目录 |
|---|---|---|---|
| section | rerank | 原始 BGE + BM25 → RRF → Cross-Encoder | artifacts/indexes/section/rerank |
| struct | rerank | 同上，使用 struct 语料 | artifacts/indexes/struct/rerank |
| section | finetuned | 微调 BGE，直接保留 Dense 排名 | artifacts/indexes/section/finetuned |
| struct | finetuned | 同上，使用训练时的 struct 表示 | artifacts/indexes/struct/finetuned |

两个开关在创建 pipeline 时选择，模型和索引启动时加载一次。应用可同时持有多个 pipeline，再按请求选择实例。微调模型在 struct 上训练，支持 section 推理不代表 section 已达到报告中的质量。

section 保留按 `id_sec` 分组规则。struct 按原 span 顺序合并：相邻 span 的 section 相同或标题相同就留在一组；不全局合并非相邻的同名标题。保留训练时标题路径、正文格式、span 映射和 `doc::structure_N` ID。

## 1. 环境与资源

所有命令在仓库根目录执行。激活含 PyTorch 的环境；AutoDL 保留镜像自带 CUDA PyTorch：

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python -m pip freeze | python -c 'import sys; print("".join(s for s in sys.stdin if s.lower().startswith(("torch==", "torchvision==", "torchaudio=="))), end="")' > /tmp/retrieval-torch-constraints.txt
python -m pip install -r src/retrieval/requirements-service.txt -c /tmp/retrieval-torch-constraints.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
```

依赖冲突时选择兼容环境，不删除约束强行替换 Torch。本次本地用原 `.venv_acceptance`（PyTorch2.6.0），从外层 `team_integration/.deps` 复用 python-dotenv；新环境按 requirements 安装即可。

原始数据：`multidoc2dial/multidoc2dial_doc.json`；也可用 `--documents /实际路径/multidoc2dial_doc.json`。

默认模型：

- 原始 BGE：`local_models/bge-base-en-v1.5`；不存在时使用 ID `BAAI/bge-base-en-v1.5`。
- Cross-Encoder：`local_models/ms-marco-MiniLM-L6-v2`；不存在时使用对应 Hugging Face ID。
- 微调：`models/bge-base-en-v1.5-struct-ft/checkpoint-2000`。通过 Git LFS 取回完整权重，不能只有 pointer。

可用 `--embedding-model /模型目录`、`--reranker-model /模型目录` 覆盖。构建和查询须保持相同路径及配置。本地模型齐全时可设置 `HF_HUB_OFFLINE=1`，下载时不要设置离线模式。

## 2. 自动化回归

正式模块统一从仓库根目录运行测试：

```bash
PYTHONPATH=src python -m pytest -q tests src/optimization_v2/tests src/retrieval/tests src/retrieval/integration_demo/tests
```

重命名后回归：主服务与优化87项、检索54项、独立demo6项，合计147项。覆盖切分、配置、直接 Dense 排名、历史输入、指标边界、索引错配拦截、生成引用和优化流程。

## 3. 小规模真实模型验收

```bash
python -m retrieval.scripts.validate_profiles \
  --documents multidoc2dial/multidoc2dial_doc.json \
  --max-docs-per-domain 1 \
  --index-root artifacts/indexes/acceptance \
  --output reports/retrieval_profiles/local_validation.json
```

按原始顺序取每领域第一份真实文档，共4份。分别建立并加载4套索引，验证 Top10 不重复、切分 ID 正确、微调链路无 BM25/RRF/重排，以及完整原生服务的生成/评价引用。已存在的索引校验后复用；换语料或模型时使用新 `--index-root`。

预期最后输出 `All four profiles passed`。本次 section103块、struct35块，4种组合全部通过。编码器与 Cross-Encoder 使用真实模型，下游生成/评价使用 MockLLM；这是集成验收，不替代 Hit/Recall 质量评测。去掉 `--max-docs-per-domain 1` 可运行全量验收，CPU 建索引耗时较长。

## 4. 建立正式索引

例如 struct + 微调：

```bash
python -m retrieval.service build --segmentation struct --chain finetuned \
  --documents multidoc2dial/multidoc2dial_doc.json --device cpu
```

其他组合修改两个开关即可。AutoDL 用 `--device cuda`，可一次建立全部组合：

```bash
for segmentation in section struct; do
  for chain in rerank finetuned; do
    python -m retrieval.service build \
      --segmentation "$segmentation" --chain "$chain" \
      --documents multidoc2dial/multidoc2dial_doc.json --device cuda
  done
done
```

正式与 acceptance 索引隔离。构建拒绝覆盖已有目录，更新语料/模型时指定新 `--index-root artifacts/indexes/v2`，完成后调用方切换到该目录。先写临时目录，全部成功后才发布。

加载校验切分、语料指纹、模型名、前缀、维度及本地模型内容指纹。同一路径换权重或 tokenizer，也须重建索引。旧 section 索引可继续使用远端模型 ID；没有内容指纹的本地模型旧索引须重建。

## 5. 检索与完整服务

仅检索（正式索引）：

```bash
python -m retrieval.service search \
  --segmentation struct --chain finetuned \
  --query "User: How do I replace a lost driver's license?" --top-k 10
```

完整服务（规划、检索、生成、评价/优化）：

```bash
python -m retrieval.native_main "How do I replace a lost driver's license?" \
  --segmentation struct --chain finetuned --top-k 10 --llm mock \
  --output results/struct_finetuned.json

python -m retrieval.native_main "How do I replace a lost driver's license?" \
  --segmentation section --chain rerank --top-k 10 --llm mock \
  --output results/section_rerank.json
```

使用已生成的小样本索引时，加 `--index-root artifacts/indexes/acceptance`。真实生成使用 `--llm deepseek`，需要已有 `DEEPSEEK_API_KEY`；本次未发送真实 API 请求。

微调初次检索拼接最近4个可见 turn 和 User/Agent 标签，由 Dense 添加 query 前缀；关闭子查询扩展，直接保留 cosine 排名。控制器后续仍可改写查询或扩大 Dense 候选，不切换为 BM25/重排。重排链路保留规划查询、Dense100/BM2550、RRF k30、候选50及原有 Cross-Encoder 策略；加载失败报错，不自动降级。

旧的 `native_main --artifact-dir ... --mode rrf|full` 仍可用。新 `--chain` 优先于旧 `--mode`。输出新增 `segmentation`、`retrieval_chain` 和相应 backend 标识。

## Python 接口

```python
from retrieval.profiles import profile_config
from retrieval import HybridRetrievalPipeline
from retrieval.schemas import RetrievalRequest

config = profile_config(segmentation="struct", chain="finetuned")
pipeline = HybridRetrievalPipeline.from_artifacts(
    artifact_dir=config.artifact_dir, config=config, device="cpu")
evidence = pipeline.retrieve(RetrievalRequest("User: Replace my license", top_k=10))
```

仍输出现有 `Evidence` 契约。微调链路的兼容字段 `rerank_score` 存 Dense cosine、`rrf_score=0`；重排链路存 Cross-Encoder 分数，跨链路分数不可直接比较。

## 6. 测量 struct 微调链路的检索指标

使用第4节建立的正式全量索引，不能用 acceptance 小样本索引评价完整测试集。先用 dev 前20条检查评测链路：

```bash
python -m retrieval.scripts.evaluate_profile \
  --segmentation struct --chain finetuned \
  --samples retrieval_finetune/struct_finetune/data/dev.json \
  --limit 20 --device cpu --output results/struct_finetuned_dev20.json
```

再对冻结 test400 执行最终评测：

```bash
python -m retrieval.scripts.evaluate_profile \
  --segmentation struct --chain finetuned \
  --samples retrieval_finetune/struct_finetune/data/test.json \
  --device cpu --output results/struct_finetuned_test400.json
```

输出 Hit/Recall/MRR/nDCG@5、@10，平均及P95延迟，各领域汇总、逐query排名和输入/索引指纹。query 与 gold 原样使用冻结输入；不使用领域过滤，不调用 Planner/LLM，不做 BM25、RRF 或重排。标签来自下一条 agent 的引用 span 映射到 struct passage，与训练对照实验口径一致。测试集一个 gold 时 Hit=Recall；不要用测试结果反复选择参数。

这些冻结数据属于运行资源，不在训练代码 Git 中；本机已补齐，新机器需从原 AutoDL 包复制或运行 prepare。不要使用旧 `scripts/evaluate_retrieval.py` 的 section 标签直接评价 struct。

历史正式 test400 中 B 的 Hit/Recall@10 为78.25%，MRR@10约0.5286，nDCG@10约0.5904，这是 AutoDL 原实验参考，不是新主服务入口已经重跑的结果。CPU服务延迟含线程池开销且硬件不同，不应直接与原 CUDA 延迟作性能对比。
