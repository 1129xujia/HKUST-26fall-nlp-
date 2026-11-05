# 统一数据契约 v1

代码来源为 src/customer_service_rag/schemas.py。所有模型拒绝未知字段，并校验赋值。
可通过 cs-rag schema --output docs/contracts.schema.json 导出机器可读契约。

## 输入与标注隔离

DialogueInput：

- conversation_id：非空对话 ID。
- current_message：本次用户问题，不包含待预测的客服答案。
- dialogue_history：按时间顺序排列的 role/text 对象；role 为 user、agent 或 system。

EvaluationExample 在上述输入外保存 sample_id、split、reference_answers、gold_citations、
gold_evidence_groups、grounding、expected_behavior 和 source。
在线代码只接收它的 dialogue 字段，不接收其他参考字段。
sample_id 在执行结束后写入 RunResult，供跨系统逐样本比较。
没有行为标注时 expected_behavior 为 null，不能默认将官方数据标注为可回答。

## 查询对象

QueryPlan 继承 DialogueInput，增加 rewritten_query、intent、domain、sub_queries。
未知意图与领域填 unknown，不能编造用户属性。
原始查询、改写查询与子查询按规范化文本去重，查询分支 ID 为 original、rewritten、sub_1 等。

D2 mock 改写只拼接历史与当前问题；不是正式的指代消解或查询改写模型。

## Passage 与检索结果

- Passage 保存 doc_id、passage_id、title、text、domain、source、span_ids、char_start、char_end。
- 引用键为 doc_id:passage_id，两种 ID 中都不能包含冒号。
- char_start 与 char_end 同时存在或同时缺省；采用 Python 字符偏移的左闭右开区间。
- 合成 passages 不伪造官方 span 和字符偏移。
- RetrievalHit 使用 passage 对象加 source、query_id、score、rank；相较初始扁平 JSON 草案，
  正文只由 Passage 统一建模，检索分数放在命中对象中。
- RetrievalCandidate 将相同 passage 的所有 hits 聚合，分别记录 fusion_score 与 rerank_score。
- 不比较 Dense 与 BM25 原始分数；RRF 为各独立结果列表贡献 1/(k+rank)。
- RetrievalResult 同时保存最终 passages、候选 candidates、原始 hits 和分支耗时。

## 回复与评估

GeneratedResponse：

- answer：非空回复。
- citations：当前提供的证据 ID。
- retrieval_round：实际检索轮次，由 Orchestrator 填写。
- status：answered、clarification_needed、insufficient_evidence、escalated 或 error。

EvaluationFeedback 的四项评分均为 [0, 1]：
relevance、sufficiency、groundedness、dialogue_consistency。
decision 为 pass、retry、clarify、escalate，并保留 missing_information、suggested_action、reason。
额外提供 clarification_question 和 suggested_queries，分别用于澄清与补充检索。
D3 已实现词面 mock 评估与阈值 Policy，正式语义评估算法尚未接入。
原始反馈和 Policy 后的反馈分别记录，评分不能替代独立离线质量评价。

## 优化动作

OptimizationAction.action：

- rewrite_query：更新 rewritten_query 后重新检索。
- add_sub_queries：增加子问题后重新检索。
- adjust_retrieval：通过 retrieval_options 更新检索参数后重新检索。
- regenerate：使用已有证据重新生成，不增加 retrieval_round。
- clarify：输出 clarification_question，等待新的用户输入。
- escalate：输出人工转接包。

优化动作由编排器重验后交给动作执行器处理。
每类动作只允许自己的字段，必需字段不得缺失或为空白；禁止通过动作开启尚未实现的 reranker。
重复子查询和相同查询/参数会被识别，无实际变化时停止循环。

## 运行与用量

RunState 和 RunResult 保存 feedback、actions、retrieval_options、retrieval_history 与 candidate_history。
最终回复与候选历史分开，澄清/转接时不把被拒绝的草稿当作最终回答。
iteration 记录已发起的评估调用，retrieval_round 记录检索调用，retry_count 记录已执行的检索/生成重试。
TraceEvent 通过 event 区分 transition 和阶段结果，elapsed_ms 表示请求开始后的累计毫秒数。
各检索分支耗时另行记录。mock_scenario 仅在显式脚本模式下填写，普通样例为 null。
TokenUsage.source 为 estimated、reported 或 unavailable，无法获得用量时保留 null。

EscalationPackage 包含原因、对话、当前查询、跨轮证据、缺失信息、评估反馈和已尝试动作；
delivered_to_human 保持 false，不声称已联系真人。

## Gold evidence 表达

- gold_citations 是可接受证据 ID 的扁平集合，便于直接检查引用与简单场景。
- gold_evidence_groups 的每个内部列表表示一项必需证据的可替代 passages。
- 同一个 grounding span 落在两个重叠窗口时，可以写成一个含两个 ID 的组，命中任一即可覆盖该项证据。
- 多来源问题需要分别命中多个组；不能把所有可替代窗口都当作独立必需证据。
- 合成样例的多个 gold_citations 默认表示多个必要证据，未显式提供分组时按单元素组处理。
- 官方 grounding 到分组的映射在 D4 完成，必须保留原始 doc_id 与 span_ids。

## 配置覆盖规则

顺序：default.yaml → 指定配置 → 实验基础配置与 profile（如指定）
→ .env / 系统环境变量 → 显式 CLI 覆盖。

- 系统环境变量优先于 .env。
- 支持的开发环境变量为 CS_RAG_SEED 和 CS_RAG_INDEX_DIR。
- 所有语料、索引和结果相对路径都基于项目根目录。
- 模型 backend 目前只接受 mock/hash，避免误把正式模型名称填入后仍运行 mock。
- 评分阈值只是初始开发值，不能称为已在验证集校准。
- quality、optimization、escalation 可通过 d3.yaml 启用 mock 实现；rerank 与初始 decompose 仍明确报错。
- 配置加载、索引构建、生成结果和运行日志不包含密钥。
