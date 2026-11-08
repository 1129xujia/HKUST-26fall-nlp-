# Struct fine-tuned BGE checkpoint

正式训练在验证集上选出的 `checkpoint-2000`，基于 `BAAI/bge-base-en-v1.5` 全参数微调。这里不是两步 smoke 的产物。原始 checkpoint 保留在 `checkpoint-2000/`，其自动生成的 README 仅供结构参考；实验信息以本文件和 `experiment/` 的实际记录为准。

## 下载和加载

权重使用 Git LFS 管理。先安装 Git LFS，再在仓库根目录执行：

```bash
git lfs install
git lfs pull --include="models/bge-base-en-v1.5-struct-ft/checkpoint-2000/model.safetensors"
```

在配置好 sentence-transformers 的环境中：

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("models/bge-base-en-v1.5-struct-ft/checkpoint-2000", device="cpu")
query = "Represent this sentence for searching relevant passages: " + "User: How can I renew my license?"
query_vectors = model.encode([query], normalize_embeddings=True)
passage_vectors = model.encode(["Document title\nSection title\nPassage body"], normalize_embeddings=True)
scores = query_vectors @ passage_vectors.T
```

模型输出768维，CLS pooling，最大512 tokens，使用归一化向量。query 必须加上述前缀；passage 无前缀，正式系统使用标题路径与正文组成的 `index_text`。原实验 query 是最近4个可见 turn，带角色标签。

将此模型接入服务时，用它重新生成文档向量；不要混用原始 Dense 的索引。此提交只交付模型，不切换服务默认配置。

## 训练和结果

- MultiDoc2Dial struct 语料4110块；去重训练输入20607条；dev160、test400。对话划分互斥，文档跨划分共享，属于域内评测。
- 2000步，batch8，全12层，AdamW lr2e-5，weight decay0.01，warmup10%，梯度裁剪1.0，随机种子20261108。
- 多正例 logsumexp 对比损失，scale20，同文档未标注候选 mask；仅使用 dev 选择 checkpoint。
- test400：原始 Dense Hit@10 59.75%；微调 Dense 无重排78.25%；原始混合召回66.25%；原始混合召回加 Cross-Encoder74.50%。每 query 一个 gold，Recall@10 与 Hit@10 相等。
- 微调 Dense 与原始混合召回加重排的差值3.75个百分点，对话聚类 bootstrap 95%区间[-0.72, 8.17]，不构成稳健优越性的证据。

完整参数、选模、训练耗时和对照统计保存在 `experiment/`；其中路径是原 AutoDL 环境的记录，不是当前仓库路径。

## 完整性

`checkpoint_manifest.json` 保存原始 checkpoint 每个文件的大小与 SHA256。权重 SHA256：

```text
6b800fc06ab42182b69e8de46770c48e0b2940833d4c8e99b1ae8d34e6390aa6
```

权重约418 MiB，包含完整模型参数，不包含训练数据或优化器状态。该 checkpoint 用于推理和评测；没有保存优化器状态，不能精确恢复原训练进度。基础模型及数据的使用条件仍须遵循各自上游许可。
