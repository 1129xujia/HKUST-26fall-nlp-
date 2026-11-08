# Part 1 / Part 2 / Part 3 独立融合 Demo

该目录只依赖 Part 2 暴露的 `retrieve(RetrievalRequest) -> list[Evidence]` 契约：

- `part1/`：规则式查询规划、上下文补全、领域识别和子查询生成。
- `../retrieval/`：正式混合检索实现，不反向依赖 Demo。
- `part3/`：可追溯的抽取式回答和引用质量检查。
- `orchestrator.py`：唯一的装配层；领域过滤无结果时自动无领域重试一次。

快速展示（只需 BM25）：

```bash
.venv_acceptance/bin/python -m retrieval.integration_demo.run_demo \
  "How do I renew a driver license?" --mode bm25 \
  --output artifacts/demo_bm25.json
```

完整展示：

```bash
HF_HOME="$PWD/.model_cache" .venv_acceptance/bin/python \
  -m integration_demo.run_demo \
  "How do I renew a driver license?" --mode rrf \
  --output artifacts/demo_full.json
```

三个部分通过窄接口连接，因此 Planner、Retriever、Generator 均可单独替换和单元测试。

对话上下文测试：

```bash
HF_HOME="$PWD/.model_cache" HF_HUB_OFFLINE=1 .venv_acceptance/bin/python \
  -m integration_demo.run_demo "What documents do I need for it?" \
  --history integration_demo/examples/renewal_history.json --mode rrf \
  --output integration_demo/output/context_case.json
```

离线展示页：

```bash
.venv_acceptance/bin/python -m retrieval.integration_demo.render_report
```

该命令只读取已经通过的 `artifacts/acceptance/` 结果，输出 `integration_demo/output/demo_report.html`。页面包含查询规划、可展开的证据、带引用回答和检索消融指标。

这是接口融合 Demo。Part 1 使用可替换的规则实现，Part 3 使用抽取式生成；Evaluator 检查引用来源对应关系。无证据时安全返回找不到证据的回答，领域过滤无结果时至多放宽领域重试一次。复杂的语义评价和反馈驱动优化仍待正式 Part 1/3 接入。
