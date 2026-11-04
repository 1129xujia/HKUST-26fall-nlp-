# 独立改进实验

正式检索模块不导入此目录。实验复用其公共接口，产物放在 `experiments/output/`，不会覆盖正式语料、索引和验收结果。

早期实验只测试了“最后一条用户消息”，现已由系统化重排优化实验取代。正式诊断入口支持固定候选、query/候选深度/passage 格式、Gold 升降和可恢复评分缓存：

```bash
export HF_HOME="$PWD/.model_cache"
export HF_HUB_OFFLINE=1
python -m experiments.optimize_reranker \
  --limit 80 --depths 50 100 200 \
  --query-modes current_first_user_history
```

当前结论与全部正负结果见项目根目录 `RERANKER_OPTIMIZATION_LOG.md`。以下内容保留为早期实验历史。

```bash
export HF_HOME="$PWD/.model_cache"
export HF_HUB_OFFLINE=1
python -m experiments.rerank_query_ablation
```

该实验读取已经通过的完整验收所记录的同一批样本，固定候选召回与其他配置，仅替换重排查询策略。结果保存到 `experiments/output/latest_user_rerank/evaluation.json`。

是否提升以真实测量为准。此实现未作为生产默认策略；多轮对话中当前一句可能只有 “yes” 或代词，丢弃上下文未必合理。后续应由 Part 1 生成独立、完整的检索/重排 query，再进行全量验证。

本轮实际结果见 [RESULTS.md](RESULTS.md)：同样本 Hit@10 从原 Full 的 23.75% 提升到 30.00%，但仍低于 RRF 的 33.75%。
