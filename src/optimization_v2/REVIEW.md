# optimization_v2 流程、提示词与分流审查

本文件保留修复前的审查记录。2026-11-06 统一入口已修正评价契约、矛盾反馈放行、证据足够时的改答、正文引用校验和默认 DeepSeek 依赖，实际联调见 [统一入口说明](../../docs/unified_integration.md)。

日期：2026-11-06。实际目录名为 `main/src/optimaization_v2`（保留现有拼写）。本次只审查该模块及它实际调用的共享评价器、Schema和重排代码，没有修改这些源码，也没有发送真实模型请求。

## 审查结论

四分支控制器、共享预算、失败时不发布被拒草稿、缓存与并发隔离已有基础。应优先修正评价契约、放行条件和失败动作的后续选择，再优化提示词长度及检索参数。当前尚无正式数据集上的四分支质量收益实验，不能把脚本演示当作提升证据。

验证：Python 3.12.7 / Pydantic 2.8.2 / NumPy 1.26.4，已有60项离线检查全部通过。补充了7组使用模拟组件的反例/观察，结果见 [offline_cases.json](offline_cases.json)。这些是控制流程与契约检查，不是模型质量测量。核验时未读取实际密钥、未调用真实API。

## 1. P1：默认重新评价无法输出路由优先使用的v2诊断

位置：`adapters.py:191–193`、`customer_service_rag/quality/evaluator.py`、`customer_service_rag/prompts/evaluation.jinja2:3–7`、`schemas.py`的Feedback。

`MainEvaluatorAdapter`先交给旧EvaluatorAgent解析成EvaluationFeedback，再转为Feedback。旧提示词只有四分、decision及自由文本建议，没有failure_types/evidence_sufficient/preferred_retriever；旧Schema还会拒绝这些额外字段。

已复现：模拟模型返回完整v2诊断，上述三个字段被ValidationError拒绝；如果按旧提示词返回，三个字段分别落到空列表/None/None。因此初始调用方可以手动传入v2诊断，但默认重新评价路径不能补充这类诊断。旧suggested_action仍可触发分支，不应误述为所有分支完全不可用。

建议：为优化器增加专用v2评价器，直接验证Feedback并使用匹配的Schema/提示词；保留旧适配器作为兼容选项，避免直接改变其他系统共享的旧评价器。

## 2. P1：pass可以压过明确的负面诊断

位置：`router.py:8–26,33–41`。

apply_policy检查四项分数、回答状态和引用列表，不检查pass与failure_types/evidence_sufficient的矛盾；route又先处理pass，再检查missing_user_information。

已复现：四分0.9、有效引用，同时decision=pass、evidence_sufficient=False、failure_types=[missing_user_information]，实际分流仍为pass。

建议：在放行前验证反馈一致性；证据明确不足或有未解决的诊断时不能pass，缺少必要用户信息应澄清。终止型clarify/escalate可以保持优先；不能只调整提示词而保留本地放行漏洞。

## 3. P1：把回答不完整当证据不足，并过早结束整次优化

位置：`router.py:61–72`、`controller.py:190–193`。

旧评价提示词要求分数描述候选回答，因此回答遗漏条件可能得到低sufficiency，但这不表示现有证据也不足。由于默认评价没有evidence_sufficient/answer_defect，新路由会把这种低分导向adjust_hybrid；即使旧评价器明确suggested_action=regenerate，也会被覆盖。随后只要新上下文和query相同，控制器立刻no_context_change，剩余重试预算没有用于另一种有依据的修正。

已复现：证据已有30天期限，回答只漏写期限。实际选adjust_hybrid，生成器调用0次，retry_count=1/max_retries=2，结束原因为no_context_change。显式suggested_action=regenerate也被改到adjust_hybrid。

建议：明确区分“证据是否足够”和“候选回答是否完整”；证据足够时先改答。无效检索动作应标记该动作无效，在剩余共享预算内选择另一个有诊断依据且未重复的动作；确实缺证据时继续保持澄清/停止，不能盲目重生成。保留总预算与不发布坏草稿的现有保护。

## 4. P2：正文临时引用与结构化引用列表可以不一致

位置：`adapters.py:124–141`、`controller.py:valid_citations`。

生成提示词要求使用E1/E2，但只转换citations数组，answer中的临时ID没有同步转换；本地校验仅验证数组成员。

已复现：只有一条证据，返回answer="Bring ID [E2]"、citations=[E1]，转换后数组成为[p::1]，正文仍为[E2]，本地引用结构检查为True。后续模型评价可能发现问题，但没有确定性保证；不能称为已经无条件发布错误答案。

建议：先明确产品引用协议。若正文需要引用，则要求逐事实标注并核对正文ID集合与数组一致，正文/数组都做同一映射；若只用结构化引用，则明确禁止正文残留临时ID。生成器与评价器看到的ID应一致，避免混用E#、原始ID、Base64 ID。

## 5. P2：当前接口下“使用用户历史重排”通常退化为identity

位置：`retrieval_adapter.py:103–108`及重排调用、`retrieval/retrieval/reranker.py:78–87`。

适配器把rewritten_query传给检索和重排，没有把QueryPlan.dialogue_history单独传给Cross-Encoder。current_first_user_history策略只会从query字符串中提取字面User:行；普通独立query没有这种行时就原样返回。

已复现：历史含New York条件，两个策略传给重排模型的主query都为permit application deadline，未含该历史文本。若rewrite本身保留了条件，信息可能仍在query里；此处不是断言所有改写都丢失历史，而是该配置并未主动使用历史。

建议：区分recall_query与rerank_query。召回继续用独立query；重排按明确格式组合当前问题和允许的历史用户轮次。记录最终发送给重排器的字符串，再做identity/历史/子查询消融，不能只看策略名称。

## 6. P2优化项：生成提示词重复携带每段证据正文

位置：`adapters.py:114–132`、`budget.py:49–51`。

payload同时包含passages和evidence，再整体序列化到user提示词，两处都包含完整text；两套ID也提高了模型理解成本。当前预算按UTF-8字节保守预留，重复文本会增加预算拒绝概率。

已核对模拟输入：同一段56字节正文两处各出现一次。不能据此宣称整个prompt可精确减半。

建议：模型可见输入只保留一份精简证据（稳定别名、标题、正文、必要条件），其余分数/调试元信息保留在运行记录。如果兼容客户端还依赖payload.passages，可区分兼容payload与真正序列化到提示词的输入，避免破坏旧客户端。

## 7. P2：已知用量的异常完成也被账本记成未知

位置：`budget.py:52–58`、`deepseek.py:usage记录及finish_reason检查`。

DeepSeek客户端已经保存有效provider用量后，若因截断/无效完成抛异常，BudgetLedger仍无条件把used_tokens设为None。

已复现：上游已用10，提供方返回16，finish_reason=length，客户端有16的已知用量，但账本总量变为null。它没有虚报零，也会停止；问题是丢失了本可追踪的已知成本。

建议：区分携带已知usage的完成错误与用量不明的网络/取消错误；前者计入已知用量后停止，后者保留未知标记。继续保持不自动重发。

## 提示词具体调整

### 评价器：最优先

先让解析器接受v2 Feedback，再调整提示词。提示词需明确定义：

- evidence_sufficient判断“证据能否回答原始问题”，不因候选漏写就变成False。
- sufficiency/完整性判断候选是否覆盖问题中的要求；证据足够但答案漏条件→answer_defect→regenerate。
- 关键用户条件缺失→missing_user_information→clarify；不把可从文档补充的事实当作要用户提供的信息。
- 对失误给出具体claim/遗漏条件及证据ID，不能只写“答案不好”。可先放在reason；如新增结构化issues字段，需同步扩Schema。
- suggested_action使用明确枚举，禁止自由造动作名。
- pass要求没有未解决问题、证据足够、引用有效；与本地Policy约束一致。
- 评价器只有最终证据时，不能可靠断言“完整候选池没有召回”或某检索分支更好。区分召回/重排应增加候选池诊断，证据不足时不强行给preferred_retriever。
- 现有四项0.9＋pass的示例存在输出锚定的可能；这是待验证风险，不是已测到的因果结果。可改用Schema和字段定义，并用失败/澄清反例验证。

### 生成器

规定“历史/旧回答/反馈不是新的事实依据”；新事实只来自证据。修订应同时满足原问题的全部仍有效条件，不能机械照抄评价器建议。明确支持的状态及引用协议。只发送一份证据正文。

### 查询改写

强调最小必要改写、保留否定/日期/数量/地区/资格条件；用户身份和限制只来自原问题及可见历史，旧草稿与检索片段仅提供检索线索。正确且完整的问题不强行改写。评价器suggested_queries快路径同样遵守约束；当前结构/去重校验不等于语义保真。

## 建议的分流原则

1. 处理合法终止决策和已无矛盾的通过。
2. 缺必要用户条件：针对性澄清。
3. 明确query误解且需要改检索：最小改写，保留原始问题和约束。
4. 证据足够、回答有缺陷：仅修订回答。
5. 证据不足：先诊断候选池。池内已有所需信息→重排/重选；池内缺失→有依据地扩召回或调整query；只有分支证据支持时才调权。
6. 某动作无变化：记录动作失败，在共享预算内选另一个有依据的动作；无可行动作则停止，不发布坏草稿。

现有扩大配额、增加上下文、改变融合权重并同时扩大重排候选的规则会混合多个因素。修正控制流程后，应先用小开发集逐项消融，记录修复成功率、误放行率、无效检索比例、模型调用数、Token和耗时；固定初始回答和评价起点，再比较分流版本。当前没有数据支持具体新权重或阈值最优。

## 建议实施顺序

先处理1/2/3（反馈契约→本地放行→无效动作处理），再处理引用与真正的重排输入，随后去重prompt并完善失败成本记账。保留共享预算、最大两次修正、严格候选守恒、可信缓存键和失败草稿隔离，不需要重写整个框架。

审查文件哈希见 [source_hashes.json](source_hashes.json)。原有60项检查通过不表示上述反例已被修复。

---

我运行了现有 60项离线测试，全部通过；另外补充了7组模拟案例，复现了以下问题。
1. 评价器无法返回新分流所需的诊断字段。
   默认评价仍先经过旧 Schema，`failure_types`、`evidence_sufficient`、`preferred_retriever` 会被拒绝。因此重新评价主要依赖旧的 suggested_action，无法使用新增诊断。应增加优化器专用的 v2 评价器，同步提示词和解析契约。[调用位置 (line 191)](C:/Users/Astrella/Desktop/codex-multi-agent/main/src/optimaization_v2/adapters.py:191)
2. “通过”可能压过明确的负面诊断。
   已复现：四项分数均为0.9，同时反馈写了“证据不足、缺少用户信息”，实际仍走 pass。放行前应检查字段是否矛盾，不能只检查分数和引用列表。[分流位置 (line 33)](C:/Users/Astrella/Desktop/codex-multi-agent/main/src/optimaization_v2/router.py:33)
3. 回答遗漏会被当成检索不足，而且可能提前结束。
   例如证据已有“30天”，回答漏写期限：即使建议动作是 regenerate，低 sufficiency 仍可能把它改成扩检索。检索上下文没变后，直接 no_context_change 结束，生成器一次都没调用，剩余预算也没用。应区分“证据不足”和“回答不完整”，并把一次动作无效与整个请求停止分开。[停止位置 (line 190)](C:/Users/Astrella/Desktop/codex-multi-agent/main/src/optimaization_v2/controller.py:190)
提示词建议分别调整：
角色	重点
评价器	明确定义证据充分性与回答完整性；输出明确诊断和动作枚举；不能仅凭最终几条证据猜测是召回还是重排的问题
生成器	反馈只是修订建议，不能作为事实依据；统一正文引用与引用列表。目前正文 [E2]、列表 E1 的矛盾能通过本地引用检查
查询改写	强调最小改写，保留否定、日期、地区和资格条件；旧草稿、检索片段不能被当作用户背景


另外两个值得处理的点：
- 历史重排策略通常没有真正拿到历史：适配器只传 rewritten_query，普通独立query下，历史策略与 identity 的输入相同。
- 生成提示词重复发送证据正文：`passages` 和 evidence 各带一份，增加费用和预算压力。
建议先修前3项，再处理引用、重排输入和提示词去重。完整依据、反例及实施顺序已写入上文
