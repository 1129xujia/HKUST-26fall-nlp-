# 离线共享开发样例

本目录包含 12 个 passages 和 16 条对话样例，均为本项目人工编写的虚构数据。
这些内容不是 MultiDoc2Dial 数据，也不代表任何机构的真实服务政策。

- 领域标签：studentaid、dmv、ssa、va，仅用于模拟后续的数据接口。
- 12 条可依据证据回答、2 条需要澄清、2 条需要转接的场景。
- 包含省略、指代、再次申请、国际学生、多条件请求、精确编号等输入。
- expected_behavior、reference_answers、gold_citations 属于离线标注，不能作为在线智能体输入。
- 默认单次模式只摘录首条证据；configs/d3.yaml 可启用 mock Evaluator / Policy，
  根据可见输入执行澄清与转接规则。普通流程不会读取 expected_behavior 等标注。
- D3 词面规则不等于真实质量评价，多来源与指代问题仍可能无法正确回答。
- char_start、char_end、官方 grounding 映射不在此虚构语料中伪造。

文件：

- passages.jsonl：每行一个 Passage。
- dialogues.jsonl：每行一个 EvaluationExample，其 dialogue 字段是在线输入。

所有本目录样例由项目编写，可随项目提交；官方数据仍需按数据集自身许可使用。
