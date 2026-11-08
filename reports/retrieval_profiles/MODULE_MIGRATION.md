# 正式检索模块迁移验证

2026-11-08：正式路径为 `src/retrieval/`。算法文件直接放在包内，统一使用 `retrieval.config`、`retrieval.pipeline`、`retrieval.schemas` 等导入，不保留版本后缀或重复嵌套包。

服务命令：`python -m retrieval.service`；完整主服务：`python -m retrieval.native_main`；指标：`python -m retrieval.scripts.evaluate_profile`。均在仓库根目录、`PYTHONPATH=src` 下运行。主服务、Part3、优化模块、multi_rag 调用及文档已同步迁移。

验证：

- 主服务/优化/指标87项、检索54项、独立demo6项，统一回归147项通过。
- 用新入口加载已有全量4110块 struct 微调索引，真实检索成功，无需重建索引。
- `retrieval.native_main` 使用真实微调模型和全量索引、Mock生成/评价，返回有效引用并以 `evaluation_passed` 结束。
- 指标及切分构建CLI入口可正常解析；源代码和当前说明文档无旧模块导入引用。

此次仅调整模块结构和调用路径；没有重新训练模型，没有重新生成正式质量评测指标，也没有执行 GitHub 推送。
