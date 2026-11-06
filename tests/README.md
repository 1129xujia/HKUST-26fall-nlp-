# 成员三：离线测试与交接

从仓库根目录执行以下命令。使用 Python 3.10+；首次运行先按根目录 `README.md` 安装主项目依赖。下面的测试不需要 API Key、模型权重、MultiDoc2Dial 或 Part 2 索引。

```powershell
python -m pip install -e .
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -p "test_part3_*.py" -v
python scripts/verify_part3_behavior.py
```

预期单元测试输出 `Ran 10 tests`、`OK`；mock 行为检查打印 `70/78 checks passed`，并更新 `reports/part3_behavior_mock.json`。这些数字对应当前 `mock-extractive-v1` 和仓库内固定的 18 条案例。若修改 mock、样例或判断标准，应重新运行并同步更新 README 与报告，不应强行保持旧数字。

## 各测试验证什么

| 文件 | 验证点 |
| --- | --- |
| `test_part3_planner.py` | 实际 `QueryPlanner` 接收历史、规范化领域和子查询；关闭历史时不把历史发给模型 |
| `test_part3_contract.py` | `QueryPlan` 到 `RetrievalRequest`、真实 `make_retrieval_node` 的公开边界、Evidence 顺序、双证据引用、含 `::` 的 Part 2 passage ID、空证据、非法引用拒绝 |
| `test_part3_deepseek.py` | 使用假 HTTP 响应验证 JSON 请求、用量记录、缺密钥与截断输出；不访问网络 |
| `fixtures/part3_behavior_cases.json` | 16 条现有 mini 样例和 2 条补充追问的人工预期字段 |
| `../scripts/verify_part3_behavior.py` | 调用当前 mock Planner 和生成器，记录每条查询、领域、回答与检查结果 |

行为检查的失败项**不会让脚本返回非零退出码**：它是诊断当前 mock 能力的报告，不是要求 mock 具有真实语义能力的单元测试。具体失败项见 JSON 中每条案例的 `query_checks` 与 `response.checks`。目前 8 个失败检查来自 6 条追问未被改写成独立查询，以及 `mini_004` 漏掉第二条证据的答案内容与引用。逐例语义复核与判分口径见 `reports/part3_semantic_review.md`；该复核由 Codex 完成，未经独立人工标注员复核。

## 结果的适用范围

生成阶段使用样例自带的 gold citation 找到证据，再把证据交给生成器。因此它只检查**已有正确证据时**的回答行为，不测 Part 2 召回率。关键词是否出现也只能做粗粒度诊断，不能证明事实有依据。正式模型、正式索引和 MultiDoc2Dial 的实验需要另行运行，不能把 `70/78` 写成论文中的准确率。

组员接入时只需依赖 `customer_service_rag.part3` 的公开函数、`QueryPlanner`，以及 Part 2 的 `make_retrieval_node`。质量评分、重试、澄清和转人工由成员四接入。见 `docs/part3_README.md` 的调用示例。
