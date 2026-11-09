# 现有检查用例

用例按 `planning/`、`generation/`、`models/`、`retrieval/`、`optimization/`、`orchestration/` 和 `finetune/` 分类。`fixtures/dialogue_cases.json` 保留已有的开发对话预期，不能当作正式质量标注。

这些是 `tests/` 内的分类目录；正式实现统一从 `customer_service_rag` 导入，其中检索为 `customer_service_rag.retrieval`，评价优化为 `customer_service_rag.optimization`。

本轮仅随生产代码重构维护原有用例的导入和调用，没有增加或执行测试套件。原生检索用例通过 `ProjectRetriever.from_pipeline(...)` 注入固定组件，使用正式 `RAGOrchestrator` 和标准 `RunResult`；不再有独立 native 或 mock 报告入口。

后续需要运行时，从仓库根目录安装项目和开发依赖，再执行 `python -m pytest tests`。模型协议用例使用本地假服务，不访问 DeepSeek；Dense 模型相关用例还需要检索依赖。测试结果与真实数据上的检索、回答质量实验分别记录。
