# 查询重排变体实验结果

日期：2026-10-28。与正式验收共用四领域均衡的 80 条 validation 样本，每领域 20 条。

## 设计

保持历史 query 的 Dense/BM25 召回、RRF、候选数、模型、领域过滤和去重配置不变，仅在 Cross-Encoder 打分时使用最后一条用户消息，并去掉 `User:` 角色前缀。对照使用 `artifacts/acceptance/evaluation_full.json`，实验样本列表由该文件逐条读取。

## 真实测量

| 策略 | Hit@5 | Hit@10 | Recall@10 | MRR@10 | 平均延迟 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 27.50% | 28.75% | 22.29% | 0.2398 | 57.8 |
| Dense | 27.50% | 31.25% | 24.79% | 0.2068 | 52.5 |
| RRF | 28.75% | 33.75% | 27.29% | 0.2648 | 100.3 |
| 原 Full | 21.25% | 23.75% | 17.29% | 0.1786 | 1,075.1 |
| 最后一条用户消息重排 | 30.00% | 30.00% | 24.79% | 0.2635 | 899.2 |

结果支持进一步检查 query 表示，但没有证明“去历史必然更好”：变体同时移除了角色前缀；某些短追问离开上下文后无法理解；而且 Hit@10 仍低于 RRF。因此没有把该变体推广为默认正式实现。

## 复现

```bash
source .venv_acceptance/bin/activate
export HF_HOME="$PWD/.model_cache"
export HF_HUB_OFFLINE=1
python -m experiments.rerank_query_ablation
```

结果位于 `experiments/output/latest_user_rerank/evaluation.json`，包含每条样本的查询、gold、召回 ID 和延迟。性能数据受机器负载影响，不设置必须相同的毫秒阈值。

下一项建议是保持完整语义的上下文重写，而不是直接删除历史；另做“仅去掉角色前缀”的对照，将两种变化分开，再用全 validation 验证。
