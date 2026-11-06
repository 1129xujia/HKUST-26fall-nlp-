# MultiDoc2Dial：20 条跨领域候选修订与复核说明

修订日期：2026-11-06。原始 JSON 与 Markdown 已备份到 outputs/cross_domain_20_review_20261106/。

本轮为 AI 辅助核对与修订，不是独立人工标注。全部样本保持 multi_document_candidate、packet_only、human_review_status=pending；未执行语义层面的全语料单文档捷径验证。业务事实按本地历史语料核对，不作为现行办理建议。

## 本轮修复

- 第 4–20 条的来源移除映射原先写反，共 34 处；现已按必答点证据重新生成。
- 第 1 条补足成人年龄，第 12 条明确客户为退伍军人本人，避免来源适用对象和表格分支被默认。
- 第 4、8、10 条补足申请渠道、表格或登录步骤；第 5、9、14 条补足永久性、工作激励和资格期间来源条件。
- 第 19 条明确询问 forbearance；与问题无直接关系的背景事实移入 optional_points，不能作为必答缺失扣分。
- 三条合成问题文字发生变化：001、012、019。不得将修改后的问题称为原始真实客户提问。

## 校验结果

- exact_quote_matches: 381
- samples: 20
- required_points: 84
- optional_points: 11
- source_exclusion_checks: 40

351 条原始引文全部匹配；新增引文也逐条核对原始 doc_id、span_id 和 text_sp。引文匹配不等于所有可能答案都已完成语义验证。

所有 required_points 均作为必答要点解释；optional_points 仅用于可选补充和事实核查，不应作为全任务完成的硬条件。渠道清单中的可选途径不要求逐一穷举。

## 仍待完成的事项

- 原始 36 个选定来源的全语料捷径/替代证据检查尚未完成。已找到相同正文或高度近似的版本；doc_group_id 仍待审，不能只按固定 doc_id 判正确性。
- 19 条 compound_parallel、1 条 single_goal_multidoc，主要测跨领域信息覆盖；不能直接解释成多跳推理或多 RAG 必然优于统一 RAG。
- 第 3 条只能按两个项目分别给出有条件的资格判断；所选证据不支持保证同时领取或推断抵扣关系。
- 第 20 条属于毕业后办事的并列请求，业务关联强度低于更紧密的共同目标，可单独分层报告。
- 所有条目仍需要真人语义复核、自然性判断以及冻结后的公平对照评测。

## 样本概览

| 编号 | 领域 | 类型 | 必答点 | 可选点 |
|---|---|---|---:|---:|
| 001 | dmv / ssa | compound_parallel | 5 | 1 |
| 002 | studentaid / va | compound_parallel | 6 | 0 |
| 003 | ssa / va | compound_parallel | 6 | 0 |
| 004 | dmv / va | single_goal_multidoc | 5 | 0 |
| 005 | dmv / va | compound_parallel | 4 | 0 |
| 006 | dmv / va | compound_parallel | 4 | 0 |
| 007 | dmv / ssa | compound_parallel | 3 | 2 |
| 008 | dmv / ssa | compound_parallel | 4 | 0 |
| 009 | ssa / studentaid | compound_parallel | 5 | 0 |
| 010 | ssa / studentaid | compound_parallel | 4 | 1 |
| 011 | ssa / studentaid | compound_parallel | 4 | 0 |
| 012 | studentaid / va | compound_parallel | 4 | 1 |
| 013 | studentaid / va | compound_parallel | 3 | 1 |
| 014 | studentaid / va | compound_parallel | 5 | 0 |
| 015 | ssa / va | compound_parallel | 4 | 0 |
| 016 | ssa / va | compound_parallel | 3 | 1 |
| 017 | ssa / va | compound_parallel | 4 | 0 |
| 018 | dmv / studentaid | compound_parallel | 3 | 2 |
| 019 | dmv / studentaid | compound_parallel | 5 | 1 |
| 020 | dmv / studentaid | compound_parallel | 3 | 1 |

## 001 · compound_parallel

> I'm 28 and recently changed my last name after getting married. I was born in the U.S. and have a New York driver's license. What do I need to update my Social Security card and my license?

可回答性：answerable；范围：packet_only。

### 必答点

- `cd1_p1` A U.S.-born adult who legally changes their name because of marriage should notify Social Security to obtain a corrected card and provide proof of the legal name change, such as a marriage document.（`Learn what documents you will need to get a Social Security Card | Social Security Administration#4_0` span 1,2,3,29,26,27,28）
- `cd1_p2` SSA generally requires current documentary proof and rejects photocopies, notarized copies and receipts. If legal-name-change proof does not sufficiently identify the applicant, or the name change is older than two years for an adult, an old-name identity document is needed; such an old-name document may be expired, with a conditional new-name alternative.（`Learn what documents you will need to get a Social Security Card | Social Security Administration#4_0` span 11,12,13,34,35,36,37,38）
- `cd1_p3` Social Security lists a U.S. driver license, state-issued non-driver ID, or U.S. passport as identity-document examples, subject to the source identity requirements.（`Learn what documents you will need to get a Social Security Card | Social Security Administration#4_0` span 43,44,45,47,46）
- `cd1_p4` For a New York driver-license name change, the person must visit a DMV office with a completed MV-44, a current license or six points of identity proof, and an original/certified U.S. marriage certificate or other qualifying name-change proof.（`How to change information on DMV documents#3_0` span 6,14,15,16,17,18,20,21）
- `cd1_p5` New York DMV requires a new photo for a license name change and lists a $12.50 amended license fee.（`How to change information on DMV documents#3_0` span 22,23,59,60）

### 可选背景，不作为必答扣分项

- `cd1_o1` The corrected card retains the Social Security number and displays the new name. — Useful outcome detail, but not necessary to answer what documents the customer needs.

### 证据包内移除检查

- 移除 `Learn what documents you will need to get a Social Security Card | Social Security Administration#4_0`：缺少 cd1_p1, cd1_p2, cd1_p3。
- 移除 `How to change information on DMV documents#3_0`：缺少 cd1_p4, cd1_p5。

### 本条修订

- Added explicit adult age to the synthetic question so the U.S.-born-adult SSA source applies without assuming adulthood.
- Separated the unchanged-number outcome from required identity-document requirements.

### 待确认条件

- Exact identity evidence available
- Whether the marriage proof permits SSA to verify identity without additional old-name documentation

**人工复核：pending。全语料最小来源数：未验证。**

## 002 · compound_parallel

> I'm a veteran getting ready to start college. How do I apply for GI Bill education benefits, and what do I need to do to apply for federal student aid?

可回答性：conditional；范围：packet_only。

### 必答点

- `cd2_p1` The VA guidance tells applicants first to check GI Bill eligibility and compare the benefits at their intended school.（`How To Apply For The GI Bill | Veterans Affairs#1_0` span 4,6,7）
- `cd2_p2` The VA GI Bill application preparation list includes Social Security number, direct-deposit information, education and military history, and school details.（`How To Apply For The GI Bill | Veterans Affairs#1_0` span 13,14,15,17）
- `cd2_p3` Describe at least one supported VA application route: request the application by telephone and mail it, seek in-person VA assistance, or work with the school certifying official. An exhaustive list is not required.（`How To Apply For The GI Bill | Veterans Affairs#1_0` span 20,24,25,28,30,31）
- `cd2_p4` The FAFSA guide recommends creating an FSA ID for electronic signing and using it to begin the FAFSA.（`Filling Out the FAFSA® Form | Federal Student Aid#1_0` span 15,17,18）
- `cd2_p5` The FAFSA preparation list includes Social Security number, financial and tax data as applicable, and savings or asset information.（`Filling Out the FAFSA® Form | Federal Student Aid#1_0` span 45,46,47,54,65）
- `cd2_p6` Describe at least one FAFSA filing route supported by this historical source, such as applying at fafsa.gov. The source also lists app and mailed-PDF routes; listing every route is not required.（`Filling Out the FAFSA® Form | Federal Student Aid#1_0` span 84,85,86,87）

### 证据包内移除检查

- 移除 `How To Apply For The GI Bill | Veterans Affairs#1_0`：缺少 cd2_p1, cd2_p2, cd2_p3。
- 移除 `Filling Out the FAFSA® Form | Federal Student Aid#1_0`：缺少 cd2_p4, cd2_p5, cd2_p6。

### 本条修订

- Accepted source-supported alternative application channels instead of requiring an exhaustive channel catalogue.

### 待确认条件

- GI Bill program and service eligibility
- School and academic-year details
- FAFSA dependency, tax, and financial circumstances

**人工复核：pending。全语料最小来源数：未验证。**

## 003 · compound_parallel

> My husband was a veteran and recently died from a service-connected illness. We were married for six years, I haven't remarried, and I'm caring for our 10-year-old daughter. Could I qualify for both VA survivor compensation and Social Security survivor benefits?

可回答性：conditional；范围：packet_only。

### 必答点

- `cd3_p1` VA DIC is described as possible tax-free monetary compensation for a survivor of a veteran who died from a service-related injury or illness.（`Benefits For Spouses And Dependents (VA DIC) | Veterans Affairs#1_0` span 2）
- `cd3_p2` The VA DIC marriage eligibility alternatives include being married to the veteran for at least one year; six years satisfies that stated duration.（`Benefits For Spouses And Dependents (VA DIC) | Veterans Affairs#1_0` span 14,20）
- `cd3_p3` VA DIC requires evidence supporting the claim, such as military service records, doctor reports or tests, and evidence of service-connected cause.（`Benefits For Spouses And Dependents (VA DIC) | Veterans Affairs#1_0` span 12,13,28,30）
- `cd3_p4` The SSA survivor rules allow widow/widower benefits at any age for a person who has not remarried and cares for the deceased worker’s child under 16 or disabled who receives benefits on that worker’s record.（`Benefits Planner: Survivors | Planning For Your Survivors | Social Security Administration#1_0` span 22,25,26,27）
- `cd3_p5` The child’s possible survivor eligibility includes being unmarried and younger than 18 (up to 19 in full-time secondary school).（`Benefits Planner: Survivors | Planning For Your Survivors | Social Security Administration#1_0` span 39）
- `cd3_p6` Social Security survivor benefits generally depend on deceased worker credits; for children and a spouse caring for children, a special six-credits-in-three-years provision may apply even without the normal credits.（`Benefits Planner: Survivors | Planning For Your Survivors | Social Security Administration#1_0` span 4,16,17）

### 证据包内移除检查

- 移除 `Benefits For Spouses And Dependents (VA DIC) | Veterans Affairs#1_0`：缺少 cd3_p1, cd3_p2, cd3_p3。
- 移除 `Benefits Planner: Survivors | Planning For Your Survivors | Social Security Administration#1_0`：缺少 cd3_p4, cd3_p5, cd3_p6。

### 本条修订

- 核对已完成；保留候选与待人工复核状态。

### 待确认条件

- Deceased husband’s Social Security work credits
- Whether daughter is eligible for and receives benefits on deceased worker’s Social Security record
- VA DIC service-connected cause evidence and any relevant cohabitation/separation details

**人工复核：pending。全语料最小来源数：未验证。**

## 004 · single_goal_multidoc

> I'm an honorably discharged veteran with a New York driver's license, but I've lost my DD214. How can I get another copy and add veteran status to my license?

可回答性：conditional；范围：packet_only。

### 必答点

- `cd4_p1` The historical source directs military-record/DD214 requests through milConnect and requires a Premium DS Logon account; it states that My HealtheVet or ID.me credentials do not work for this feature.（`Request Your Military Service Records | Veterans Affairs#1_0` span 1,2,3,4,5,6,7）
- `cd4_p2` The military-records request process includes requesting a personnel file, selecting the records, and sending the request.（`Request Your Military Service Records | Veterans Affairs#1_0` span 10,15,18,19）
- `cd4_p3` The designation is for qualified honorably discharged veterans and requires proof of honorable discharge such as DD-214.（`Veteran status designation on a photo document#3_0` span 1,3,7,10,11）
- `cd4_p4` The document has no additional designation fee, but standard transaction charges apply; office visit is required if upgrading to REAL ID/Enhanced.（`Veteran status designation on a photo document#3_0` span 3,4,22,24）
- `cd4_p5` For the mail route to add the veteran designation, complete MV-44 and send it with honorable-discharge proof to the DMV bureau listed in the source. The source says this transaction cannot be completed by phone or online.（`Veteran status designation on a photo document#3_0` span 28,29,30,31,32,33,34,35）

### 证据包内移除检查

- 移除 `Request Your Military Service Records | Veterans Affairs#1_0`：缺少 cd4_p1, cd4_p2。
- 移除 `Veteran status designation on a photo document#3_0`：缺少 cd4_p3, cd4_p4, cd4_p5。

### 本条修订

- Added the actual DMV submission route/MV-44 and milConnect account prerequisites, which were missing from the how-to rubric.
- Corrected reversed missing-points-without-source mapping; rebuilt after rubric changes.

### 待确认条件

- milConnect access
- Whether changing document type

**人工复核：pending。全语料最小来源数：未验证。**

## 005 · compound_parallel

> I hurt my leg during military service and now have major trouble walking. I'm living in New York. Could I seek VA disability compensation and a disabled parking permit, and what would each require?

可回答性：conditional；范围：packet_only。

### 必答点

- `cd5_p1` VA disability compensation concerns current conditions linked to military service and includes service and rating requirements in the supplied version.（`Eligibility For VA Disability Benefits | Veterans Affairs#1_0` span 5,6,8,9,12）
- `cd5_p2` The source warns that certain discharge statuses can affect VA disability eligibility; the user has not supplied their discharge status.（`Eligibility For VA Disability Benefits | Veterans Affairs#1_0` span 27,28）
- `cd5_p3` The cited NY disability-parking criteria require a qualifying mobility impairment that is permanent in nature; one listed case is severe walking limitation from an arthritic, neurological or orthopedic condition.（`Parking for People with Disabilities - The Law#3_0` span 13,14,21）
- `cd5_p4` NY parking permit/plates require suitable medical documentation, with MV-664.1 and the local issuing authority identified in the source.（`Parking for People with Disabilities - The Law#3_0` span 31,32,39,42,43）

### 证据包内移除检查

- 移除 `Eligibility For VA Disability Benefits | Veterans Affairs#1_0`：缺少 cd5_p1, cd5_p2。
- 移除 `Parking for People with Disabilities - The Law#3_0`：缺少 cd5_p3, cd5_p4。

### 本条修订

- Restored the permanent-mobility condition and removed unrelated preservice/postservice scenarios from mandatory scoring.
- Corrected reversed missing-points-without-source mapping; rebuilt after rubric changes.

### 待确认条件

- VA disability rating
- Discharge details
- Medical certification
- Local parking permit issuing authority
- Whether the mobility impairment is permanent and meets the listed parking criteria

**人工复核：pending。全语料最小来源数：未验证。**

## 006 · compound_parallel

> My father was a veteran and has just passed away. How do I arrange burial in a VA national cemetery and take care of his New York driver's license and car registration?

可回答性：conditional；范围：packet_only。

### 必答点

- `cd6_p1` The family or funeral director can contact the National Cemetery Scheduling Office to request burial.（`Schedule A Burial For A Veteran Or Family Member | Veterans Affairs#1_0` span 2,6,7）
- `cd6_p2` VA burial scheduling generally uses DD214 or discharge documentation and may require family/death evidence.（`Schedule A Burial For A Veteran Or Family Member | Veterans Affairs#1_0` span 17,18,20,21）
- `cd6_p3` NY DMV requests a death certificate copy and a photo copy of the deceased’s license/ID if available to close the ID record.（`If a family member has died#3_0` span 5,8）
- `cd6_p4` NY DMV describes possible registration refund to the estate and steps for plate surrender and executor-submitted evidence.（`If a family member has died#3_0` span 13,14,17,18,20）

### 证据包内移除检查

- 移除 `Schedule A Burial For A Veteran Or Family Member | Veterans Affairs#1_0`：缺少 cd6_p1, cd6_p2。
- 移除 `If a family member has died#3_0`：缺少 cd6_p3, cd6_p4。

### 本条修订

- Corrected reversed missing-points-without-source mapping; rebuilt after rubric changes.

### 待确认条件

- DD214 or discharge documents
- Burial eligibility
- Executor status
- Original registration and plates

**人工复核：pending。全语料最小来源数：未验证。**

## 007 · compound_parallel

> I receive Social Security retirement benefits and just moved to a new apartment in New York. How should I update my address for my benefits and my New York driver's license and vehicle registration?

可回答性：answerable；范围：packet_only。

### 必答点

- `cd7_p1` SSA retirement beneficiaries can use my Social Security to update contact information and enter the new address.（`Change of Address| Social Security Administration#1_0` span 3,4,6,8,10）
- `cd7_p3` NY DMV requires reporting residence address changes for license/ID or vehicle records within 10 days.（`How to change your address#1_0` span 3,4）
- `cd7_p4` Describe at least one supported DMV address-update route: MyDMV using the ID and document numbers from the listed NY identity document, or the mail route using MV-232.（`How to change your address#1_0` span 13,14,25,15,16,17,18,19）

### 可选背景，不作为必答扣分项

- `cd7_p2` The online address service described does not cover SSI recipients or people without a US mailing address. — SSI exceptions are context, not a mandatory answer item for this stated retirement-benefit customer.
- `cd7_p5` An address change through USPS does not itself update DMV records. — USPS caveat is useful context; a correct direct SSA/DMV answer need not introduce USPS.

### 证据包内移除检查

- 移除 `Change of Address| Social Security Administration#1_0`：缺少 cd7_p1。
- 移除 `How to change your address#1_0`：缺少 cd7_p3, cd7_p4。

### 本条修订

- cd7_p2 moved to optional context: SSI exceptions are context, not a mandatory answer item for this stated retirement-benefit customer.
- cd7_p5 moved to optional context: USPS caveat is useful context; a correct direct SSA/DMV answer need not introduce USPS.
- Completed the fragmentary identity-document context and accepted a valid update route without requiring all routes.
- Corrected reversed missing-points-without-source mapping; rebuilt after rubric changes.

### 待确认条件

- Exact move date
- Whether my Social Security and MyDMV accounts exist

**人工复核：pending。全语料最小来源数：未验证。**

## 008 · compound_parallel

> After a serious illness I can't work or drive anymore, but my New York driver's license is still current. How can I apply for Social Security disability benefits and switch my license to a non-driver ID?

可回答性：conditional；范围：packet_only。

### 必答点

- `cd8_p1` The SSA page advises applying as soon as disabled and permits online or telephone applications.（`Benefits Planner: Disability | How You Apply | Social Security Administration#1_0` span 2,3,5,6）
- `cd8_p2` SSA application preparations include medical provider names, contact details, and dates of treatment.（`Benefits Planner: Disability | How You Apply | Social Security Administration#1_0` span 13,14,16）
- `cd8_p3` A current NY driver license can be exchanged for a non-driver ID by mail or at a DMV office.（`Exchange a NY State driver license for a non-driver ID card#1_0` span 2,3）
- `cd8_p4` The source identifies MV-44 as the application for exchanging the current license for a non-driver ID; explain the chosen mail or office route and its source-supported requirements.（`Exchange a NY State driver license for a non-driver ID card#1_0` span 22,24,50,53,23,51）

### 证据包内移除检查

- 移除 `Benefits Planner: Disability | How You Apply | Social Security Administration#1_0`：缺少 cd8_p1, cd8_p2。
- 移除 `Exchange a NY State driver license for a non-driver ID card#1_0`：缺少 cd8_p3, cd8_p4。

### 本条修订

- Added the full MV-44 form citation and avoided requiring expired-license details for a customer with a current license.
- Corrected reversed missing-points-without-source mapping; rebuilt after rubric changes.

### 待确认条件

- Social Security work-credit record
- Expected duration of disability
- Application identity documents

**人工复核：pending。全语料最小来源数：未验证。**

## 009 · compound_parallel

> I'm receiving Social Security disability benefits and plan to enroll in college half-time while working a small campus job. Could the job affect my disability benefits, and can I receive federal Direct Loans as a half-time student?

可回答性：conditional；范围：packet_only。

### 必答点

- `cd9_p1` Substantial work can affect continuing SSA disability eligibility, but the planned campus job does not by itself establish that benefits will stop; earnings and applicable rules matter. Historical earnings examples need not be quoted.（`Benefits Planner: Disability | Are You Working | Social Security Administration#1_0` span 22,23,24,27）
- `cd9_p2` Recipients must promptly report returning to work or improvement in condition.（`Benefits Planner: Disability | Are You Working | Social Security Administration#1_0` span 30）
- `cd9_p3` Student Aid criteria include regular enrollment in an eligible program and at least half-time enrollment for Direct Loan Program funds.（`Eligibility Requirements | Federal Student Aid#1_0` span 4,10,11）
- `cd9_p4` Additional eligibility rules and program-specific criteria may apply.（`Eligibility Requirements | Federal Student Aid#1_0` span 23,24）
- `cd9_p5` The source also describes work incentives that can preserve cash benefits and Medicare while testing the ability to work. Do not decide this customer loses benefits solely from a substantial-earnings example.（`Benefits Planner: Disability | Are You Working | Social Security Administration#1_0` span 35,36,37,38,39,40）

### 证据包内移除检查

- 移除 `Benefits Planner: Disability | Are You Working | Social Security Administration#1_0`：缺少 cd9_p1, cd9_p2, cd9_p5。
- 移除 `Eligibility Requirements | Federal Student Aid#1_0`：缺少 cd9_p3, cd9_p4。

### 本条修订

- Added the work-incentive countercondition from the same source; historic earnings thresholds are not mandatory.
- Corrected reversed missing-points-without-source mapping; rebuilt after rubric changes.

### 待确认条件

- Campus job earnings
- Program eligibility
- Other federal student aid qualifications

**人工复核：pending。全语料最小来源数：未验证。**

## 010 · compound_parallel

> I'm retired, receive Social Security benefits, and my federal student loan payments have become hard to manage. How can I get an official letter showing my benefits and apply for income-driven repayment?

可回答性：conditional；范围：packet_only。

### 必答点

- `cd10_p1` An SSA benefit verification letter is available through my Social Security and documents retirement/other benefits.（`Benefit Verification Letter  | Social Security Administration#1_0` span 3,4,6）
- `cd10_p2` The letter may be used for income verification; the online account directs users to Replacement Documents.（`Benefit Verification Letter  | Social Security Administration#1_0` span 8,14,16）
- `cd10_p3` IDR is intended to lower payments for borrowers with federal student loans, and borrowers not yet on a plan can apply.（`Apply for Income-Driven Student Loan Repayment | Federal Student Aid#1_0` span 1,2,3,20,21）
- `cd10_p5` For a new IDR application, the source lists a verified FSA ID and directs applicants not already on an IDR plan to log in to start the application. Existing-plan changes use the relevant returning-applicant route.（`Apply for Income-Driven Student Loan Repayment | Federal Student Aid#1_0` span 17,18,20,21,22）

### 可选背景，不作为必答扣分项

- `cd10_p4` The source describes applicants who seek a lower payment or recertification and a single-session application process. — Single-session timing and recertification are context; they are not a substitute for instructions for a new application.

### 证据包内移除检查

- 移除 `Benefit Verification Letter  | Social Security Administration#1_0`：缺少 cd10_p1, cd10_p2。
- 移除 `Apply for Income-Driven Student Loan Repayment | Federal Student Aid#1_0`：缺少 cd10_p3, cd10_p5。

### 本条修订

- cd10_p4 moved to optional context: Single-session timing and recertification are context; they are not a substitute for instructions for a new application.
- Added actionable IDR login/FSA-ID steps rather than counting a process-duration description as application instructions.
- Corrected reversed missing-points-without-source mapping; rebuilt after rubric changes.

### 待确认条件

- Loan types
- Family size and income
- Existing repayment plan

**人工复核：pending。全语料最小来源数：未验证。**

## 011 · compound_parallel

> I'm 17 and my father recently died. I'm applying to college, and my mother has remarried. Might I qualify for Social Security survivor benefits, and whose information should I put on my FAFSA?

可回答性：conditional；范围：packet_only。

### 必答点

- `cd11_p1` Potential SSA survivor eligibility depends on the deceased worker qualifying through covered work/credits; this record is unknown, so do not guarantee an award.（`Benefits Planner: Survivors | Planning For Your Survivors | Social Security Administration#1_0` span 4,11,16,17）
- `cd11_p2` Unmarried children under 18, or certain older full-time school students, may qualify for survivor benefits.（`Benefits Planner: Survivors | Planning For Your Survivors | Social Security Administration#1_0` span 38,39）
- `cd11_p3` Dependent students report their own and legal parents’ financial information on FAFSA, subject to specified criteria.（`Reporting Parent Information | Federal Student Aid#1_0` span 1,2,3）
- `cd11_p4` A stepparent currently married to the reporting legal parent generally must also be included.（`Reporting Parent Information | Federal Student Aid#1_0` span 4,5,6,7）

### 证据包内移除检查

- 移除 `Benefits Planner: Survivors | Planning For Your Survivors | Social Security Administration#1_0`：缺少 cd11_p1, cd11_p2。
- 移除 `Reporting Parent Information | Federal Student Aid#1_0`：缺少 cd11_p3, cd11_p4。

### 本条修订

- Kept the insured-work condition while avoiding an obligatory detailed credit-rule recital for the general eligibility question.
- Corrected reversed missing-points-without-source mapping; rebuilt after rubric changes.

### 待确认条件

- Father’s work-credit status
- Student marital/dependency status
- Exact household and financial circumstances

**人工复核：pending。全语料最小来源数：未验证。**

## 012 · compound_parallel

> I am a veteran already using GI Bill benefits and submitted my FAFSA, but I'm transferring to another college next semester. What updates do I need to make with VA and on my FAFSA so the new school gets my information?

可回答性：answerable；范围：packet_only。

### 必答点

- `cd12_p1` The VA requires a benefits change request when transferring to a different school.（`Change Your GI Bill School Or Program | Veterans Affairs#1_0` span 4,5,6,7）
- `cd12_p2` Veterans/service members file VA Form 22-1995, available online, to change school or program.（`Change Your GI Bill School Or Program | Veterans Affairs#1_0` span 16,17,18）
- `cd12_p3` After filing FAFSA, add the new college Federal School Code so that the additional school can receive the information.（`How to Correct or Update Your FAFSA® Application | Federal Student Aid#1_0` span 44,45,46,47）
- `cd12_p4` FAFSA corrections can be submitted online using FSA ID and Make FAFSA Corrections.（`How to Correct or Update Your FAFSA® Application | Federal Student Aid#1_0` span 53,54,55,56,57,58,60,61）

### 可选背景，不作为必答扣分项

- `cd12_o1` This historical source limits the FAFSA list to ten schools at a time. — Relevant if the customer is already at the source-version limit; not essential for every school-addition answer.

### 证据包内移除检查

- 移除 `Change Your GI Bill School Or Program | Veterans Affairs#1_0`：缺少 cd12_p1, cd12_p2。
- 移除 `How to Correct or Update Your FAFSA® Application | Federal Student Aid#1_0`：缺少 cd12_p3, cd12_p4。

### 本条修订

- Added explicit veteran status to resolve the 22-1995 versus dependent 22-5495 ambiguity; made the school-count detail conditional context.
- Corrected reversed missing-points-without-source mapping; rebuilt after rubric changes.

### 待确认条件

- New college school code
- Whether FAFSA already lists ten schools

**人工复核：pending。全语料最小来源数：未验证。**

## 013 · compound_parallel

> VA recently gave me a 100% service-connected disability rating, and I still have federal student loans. What does my VA rating mean for compensation, and could it help me get those loans discharged?

可回答性：conditional；范围：packet_only。

### 必答点

- `cd13_p1` VA disability ratings express impairment severity as percentages and are used to determine monthly compensation amounts.（`About VA Disability Ratings | Veterans Affairs#1_0` span 5,6,7,8）
- `cd13_p2` A TPD discharge can relieve specified federal loan obligations, requiring an application and qualifying documentation.（`Total and Permanent Disability Discharge | Federal Student Aid#1_0` span 1,2,4）
- `cd13_p3` VA documentation of a 100% service-connected disabling condition or individual unemployability determination can support TPD discharge.（`Total and Permanent Disability Discharge | Federal Student Aid#1_0` span 14,16）

### 可选背景，不作为必答扣分项

- `cd13_p4` The application and necessary supporting documents are sent to the stated TPD servicer. — Detailed TPD submission/servicer instructions are optional because the question asks what the rating means and whether it can support discharge.

### 证据包内移除检查

- 移除 `About VA Disability Ratings | Veterans Affairs#1_0`：缺少 cd13_p1。
- 移除 `Total and Permanent Disability Discharge | Federal Student Aid#1_0`：缺少 cd13_p2, cd13_p3。

### 本条修订

- cd13_p4 moved to optional context: Detailed TPD submission/servicer instructions are optional because the question asks what the rating means and whether it can support discharge.
- Corrected reversed missing-points-without-source mapping; rebuilt after rubric changes.

### 待确认条件

- Exact federal loan type
- VA disability determination document
- Existing TPD notification

**人工复核：pending。全语料最小来源数：未验证。**

## 014 · compound_parallel

> I'm a veteran with a 20% VA service-connected disability rating and want to go back to college at least half-time for a new career. Could VA vocational rehabilitation help, and can my existing federal student loan payments be deferred while I'm enrolled?

可回答性：conditional；范围：packet_only。

### 必答点

- `cd14_p1` VA VR&E supports veterans with service-related disabilities limiting employment; the document describes at least 10% service-connected rating for veterans.（`Eligibility For VA Vocational Rehab And Employment | Veterans Affairs#1_0` span 2,3,4,6,7,8,10,12,14）
- `cd14_p2` The historical source describes a basic 12-year eligibility period tied to notice of separation from active military service or the first VA service-connected disability rating, with a possible serious-employment-handicap extension.（`Eligibility For VA Vocational Rehab And Employment | Veterans Affairs#1_0` span 15,16,19,20,17,18）
- `cd14_p3` In-school deferment allows pausing payments for borrowers attending eligible college at least half-time.（`Student Loan Deferment | Federal Student Aid#1_0` span 3,4,56）
- `cd14_p4` In-school deferment is generally granted based on school-reported enrollment, with contact and reporting steps if not automatic.（`Student Loan Deferment | Federal Student Aid#1_0` span 60,61,62,63,64,65）
- `cd14_p5` Interest may still accrue during deferment depending on loan type.（`Student Loan Deferment | Federal Student Aid#1_0` span 19,20,21）

### 证据包内移除检查

- 移除 `Eligibility For VA Vocational Rehab And Employment | Veterans Affairs#1_0`：缺少 cd14_p1, cd14_p2。
- 移除 `Student Loan Deferment | Federal Student Aid#1_0`：缺少 cd14_p3, cd14_p4, cd14_p5。

### 本条修订

- Restored the separation-notice branch that was missing from the eligibility-clock evidence.
- Corrected reversed missing-points-without-source mapping; rebuilt after rubric changes.

### 待确认条件

- Service/discharge and rating notice dates
- Employment handicap assessment
- Enrollment status and loan types

**人工复核：pending。全语料最小来源数：未验证。**

## 015 · compound_parallel

> An injury from my military service has kept me from working for more than a year. I used to work in civilian jobs too. Can I pursue VA disability compensation and Social Security disability benefits, and how are their eligibility tests different?

可回答性：conditional；范围：packet_only。

### 必答点

- `cd15_p1` VA disability eligibility addresses current injury linked to service with military service and rating requirements.（`Eligibility For VA Disability Benefits | Veterans Affairs#1_0` span 5,6,8,9,10,12）
- `cd15_p2` Certain less-than-honorable discharge statuses may affect VA disability eligibility.（`Eligibility For VA Disability Benefits | Veterans Affairs#1_0` span 27,28）
- `cd15_p3` SSA benefits require prior covered work and a medically qualifying condition preventing work for a year or more.（`Benefits Planner: Disability | How You Qualify | Social Security Administration#1_0` span 2,3,4,6）
- `cd15_p4` SSA requires enough recent covered work and uses a total-disability, not partial or short-term, standard.（`Benefits Planner: Disability | How You Qualify | Social Security Administration#1_0` span 12,13,31,32,33）

### 证据包内移除检查

- 移除 `Eligibility For VA Disability Benefits | Veterans Affairs#1_0`：缺少 cd15_p1, cd15_p2。
- 移除 `Benefits Planner: Disability | How You Qualify | Social Security Administration#1_0`：缺少 cd15_p3, cd15_p4。

### 本条修订

- Corrected reversed missing-points-without-source mapping; rebuilt after rubric changes.

### 待确认条件

- VA rating and discharge status
- Social Security work credits
- Medical assessments

**人工复核：pending。全语料最小来源数：未验证。**

## 016 · compound_parallel

> I'm a wartime veteran turning 65 and starting retirement planning. What would determine whether I qualify for a VA Veterans Pension, and when can I apply for Social Security retirement benefits?

可回答性：conditional；范围：packet_only。

### 必答点

- `cd16_p1` VA Veterans Pension is means-tested and conditioned on qualifying discharge and wartime veteran status.（`Eligibility For Veterans Pension | Veterans Affairs#1_0` span 2,7,8,10）
- `cd16_p2` VA pension wartime service length rules depend on service start and age 65 can satisfy the age branch.（`Eligibility For Veterans Pension | Veterans Affairs#1_0` span 14,16,17,19,21,22,23）
- `cd16_p3` SSA permits online retirement applications from 61 years 9 months and generally no more than four months before desired start.（`Benefits Planner: Retirement | You Can Apply For Retirement Benefits Or Medicare | Social Security Administration#1_0` span 4,14,17）

### 可选背景，不作为必答扣分项

- `cd16_p4` SSA pays retirement benefits the month after they are due. — Payment timing is useful context, but the customer asks when an application can be made.

### 证据包内移除检查

- 移除 `Eligibility For Veterans Pension | Veterans Affairs#1_0`：缺少 cd16_p1, cd16_p2。
- 移除 `Benefits Planner: Retirement | You Can Apply For Retirement Benefits Or Medicare | Social Security Administration#1_0`：缺少 cd16_p3。

### 本条修订

- cd16_p4 moved to optional context: Payment timing is useful context, but the customer asks when an application can be made.
- Corrected reversed missing-points-without-source mapping; rebuilt after rubric changes.

### 待确认条件

- VA discharge and service dates
- Annual family income and net worth
- SSA credits and desired start date

**人工复核：pending。全语料最小来源数：未验证。**

## 017 · compound_parallel

> I get VA disability payments and Social Security retirement benefits, and I just opened a new bank account. How do I move the direct deposits from both agencies to the new account?

可回答性：answerable；范围：packet_only。

### 必答点

- `cd17_p1` VA disability and pension payees can change deposit details through available VA.gov profile or eBenefits flows, depending on whether feature appears.（`Change Your VA Direct Deposit Information | Veterans Affairs#1_0` span 2,4,5,12,13,14,15）
- `cd17_p2` VA direct deposit changes require bank routing and account numbers.（`Change Your VA Direct Deposit Information | Veterans Affairs#1_0` span 16）
- `cd17_p3` SSA payees can access my Social Security and Update Direct Deposit within Benefits & Payment Details.（`Direct Deposit | Social Security Administration#1_0` span 2,3,6,15,16,17,18,19）
- `cd17_p4` SSA requires routing number, account type, and bank account number.（`Direct Deposit | Social Security Administration#1_0` span 9,12,13,14）

### 证据包内移除检查

- 移除 `Change Your VA Direct Deposit Information | Veterans Affairs#1_0`：缺少 cd17_p1, cd17_p2。
- 移除 `Direct Deposit | Social Security Administration#1_0`：缺少 cd17_p3, cd17_p4。

### 本条修订

- Corrected reversed missing-points-without-source mapping; rebuilt after rubric changes.

### 待确认条件

- New bank routing/account details
- VA.gov/eBenefits feature availability
- my Social Security access

**人工复核：pending。全语料最小来源数：未验证。**

## 018 · compound_parallel

> I lost my current New York driver's license while getting ready to file the FAFSA for college. How can I replace the license, and what license information will the FAFSA ask me for?

可回答性：conditional；范围：packet_only。

### 必答点

- `cd18_p1` NY licenses can be replaced if lost, stolen, or destroyed, but expired documents cannot be replaced via this procedure.（`Replace license or permit#1_0` span 2,3）
- `cd18_p2` NY DMV offers online replacement for qualified applicants and the cited replacement fee is $17.50.（`Replace license or permit#1_0` span 16,22,23,25,33,44）
- `cd18_p4` The FAFSA document lists a driver’s license number as requested information if the applicant has one.（`Filling Out the FAFSA® Form | Federal Student Aid#1_0` span 50,51）

### 可选背景，不作为必答扣分项

- `cd18_p3` FAFSA gathers personal information including Social Security number and related documents. — General FAFSA personal-information requirements extend beyond the requested driver-license information.
- `cd18_p5` FSA ID provides the online electronic signature route. — The FSA-ID electronic-signature route does not answer which license information FAFSA requests.

### 证据包内移除检查

- 移除 `Replace license or permit#1_0`：缺少 cd18_p1, cd18_p2。
- 移除 `Filling Out the FAFSA® Form | Federal Student Aid#1_0`：缺少 cd18_p4。

### 本条修订

- cd18_p3 moved to optional context: General FAFSA personal-information requirements extend beyond the requested driver-license information.
- cd18_p5 moved to optional context: The FSA-ID electronic-signature route does not answer which license information FAFSA requests.
- Corrected reversed missing-points-without-source mapping; rebuilt after rubric changes.

### 待确认条件

- Existing license number
- Eligibility for online replacement
- Other FAFSA identity documents

**人工复核：pending。全语料最小来源数：未验证。**

## 019 · compound_parallel

> I have a permanent mobility disability and recently had to stop working. I live in New York and have federal student loans. How can I apply for disability parking privileges and ask my student loan servicer for temporary forbearance?

可回答性：conditional；范围：packet_only。

### 必答点

- `cd19_p1` NY disability parking rules require mobility-related qualifying conditions and proof from an authorized medical professional.（`Parking for People with Disabilities - The Law#3_0` span 13,14,21,31,32）
- `cd19_p2` Medical certification on MV-664.1 and local town/city/village clerk are described for permits.（`Parking for People with Disabilities - The Law#3_0` span 39,40,42,43）
- `cd19_p3` General forbearance may be requested for financial difficulties or employment changes, but servicer discretion applies.（`Student Loan Forbearance | Federal Student Aid#1_0` span 46,47,48,51,52,54）
- `cd19_p4` Most forbearance requests require contacting the servicer and providing eligibility documentation.（`Student Loan Forbearance | Federal Student Aid#1_0` span 39,40,43）
- `cd19_p5` Under this historical source, the borrower remains responsible for interest accrued during forbearance; the source describes paying it or allowing accrual/capitalization.（`Student Loan Forbearance | Federal Student Aid#1_0` span 28,30,34,60）

### 可选背景，不作为必答扣分项

- `cd19_o1` The source says general forbearance may be granted for no more than 12 months at a time. — Useful duration context, separate from the interest consequence and request procedure.

### 证据包内移除检查

- 移除 `Parking for People with Disabilities - The Law#3_0`：缺少 cd19_p1, cd19_p2。
- 移除 `Student Loan Forbearance | Federal Student Aid#1_0`：缺少 cd19_p3, cd19_p4, cd19_p5。

### 本条修订

- Made the requested loan-relief path explicit so a forbearance-only rubric does not unfairly reject valid answers to a broader relief question; separated interest and duration.
- Corrected reversed missing-points-without-source mapping; rebuilt after rubric changes.

### 待确认条件

- Certified condition and issuing locality
- Loan type and servicer
- Financial difficulties and income

**人工复核：pending。全语料最小来源数：未验证。**

## 020 · compound_parallel

> I've just graduated college with federal student loans and want to start working as a truck driver in New York. What are the main steps to get a New York CDL, and do I need to complete student-loan exit counseling now?

可回答性：conditional；范围：packet_only。

### 必答点

- `cd20_p1` Describe the main source-supported NY CDL stages: a qualifying existing license, applying for a commercial learner permit and passing the written test, and the road-test stage.（`Get a CDL#1_0` span 1,3,9,10,11,14）
- `cd20_p2` CDL applicant must prepare with appropriate supervised practice, schedule road test, and obey waiting rule after CLP issuance.（`Get a CDL#1_0` span 44,45,46,47,48）
- `cd20_p3` Exit counseling applies on graduation, leaving school, or dropping below half-time for specified Direct/FFEL student loan borrowers.（`Exit Counseling | Federal Student Aid#1_0` span 3,4,5,6,7,8,14）

### 可选背景，不作为必答扣分项

- `cd20_p4` Exit counseling is completed in a single session, normally 20–30 minutes, and needs verified FSA ID. — Counseling duration and session/login details are context; the customer asks whether graduation triggers the obligation.

### 证据包内移除检查

- 移除 `Get a CDL#1_0`：缺少 cd20_p1, cd20_p2。
- 移除 `Exit Counseling | Federal Student Aid#1_0`：缺少 cd20_p3。

### 本条修订

- cd20_p4 moved to optional context: Counseling duration and session/login details are context; the customer asks whether graduation triggers the obligation.
- Removed unasked fee requirements from the main-steps point and made counseling logistics optional.
- Corrected reversed missing-points-without-source mapping; rebuilt after rubric changes.

### 待确认条件

- Current license class
- Specific loan type
- FSA ID status

**人工复核：pending。全语料最小来源数：未验证。**
