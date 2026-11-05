# D3 可控 mock 场景

d3.yaml 保存人工编写的对话输入和模拟服务返回值，用于演示控制流程。
这里不保存数据集标准答案、gold citations 或 expected_behavior。

使用 cs-rag scenarios 查看场景，用 cs-rag demo --scenario all 运行全部场景。
场景会显式注入 ScriptedEvaluator、ScriptedOptimizer 和 ScriptedGenerator，
并在结果中记录 mock_scenario。普通 demo / run 不会读取这些脚本。

脚本中的 pass 或评分只表示模拟服务返回了该值，不表示真实回答质量达标。
Policy 仍会执行阈值和有效回答检查；脚本不能绕过候选引用校验、重试限制或预算。
rewrite-query / add-sub-queries 为了完整展示动作执行，显式关闭无新证据停止；
其他场景保留默认停止规则。retry-limit 使用同一批证据连续重新生成。
