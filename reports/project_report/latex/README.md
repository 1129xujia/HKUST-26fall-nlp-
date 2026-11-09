# 项目实验报告交付说明

这是一份英文项目总报告，使用官方 ACL 双栏模板。入口为 `main.tex`，参考文献为 `references.bib`。最终 PDF 共 8 页，包含参考文献和 Limitations；已逐页检查排版。

## 本次完成的内容

- 引言：在线客服的对话依赖、复合需求与证据条件问题。
- 相关工作：RAG、MultiDoc2Dial、对话改写、迭代检索与模型反馈。
- 方法：规划器、引用式生成、评价器、优化动作、预算与停止策略。
- 实验设置：数据划分、三个独立实验群体、模型配置和指标口径。
- 非检索结果：1,000 条验证集结果，以及 20 条跨领域样例的 A/B/C 比较。
- 讨论：覆盖与成本、条件错误、转接造成的信息损失、自动评分不稳定性。
- 结论与局限。

队友的 `retrievalreport_en.docx` 已按方法、结果和讨论整合到总报告，原始 DOCX 未改动：

- `sections/03_retrieval.tex` 为独立检索方法小节，含 section/struct、BGE、BM25、RRF 公式、Cross-Encoder 与 Dense 微调。
- `sections/05_retrieval_existing.tex` 为检索结果小节，含原报告三张表；400 条对照保留 Hit/Recall、MRR、nDCG、Hit@50 和查询延迟。
- `sections/07_discussion.tex` 的检索讨论说明分块粒度、训练重编码与查询时重排的成本取舍。

2026年11月9日整合修订版还将检索贡献写入摘要、引言与结论。初版将方法压进 Results 且省略了 nDCG，本修订已恢复。统计口径和原始实验数值未改变，也未新增模型运行。

同日微调链路核对修订明确区分两种配置：`configs/native.yaml` 显式选择 struct + fine-tuned Dense，关闭 BM25、RRF、Cross-Encoder、子查询和额外语义去重；`configs/deepseek.yaml` 保留 MiniLM 混合检索及重排，历史 1000 条与 20 条结果来自该后端。原生微调配置不是无参数运行时的全项目默认。图 1 现展示原生微调路径，不能将旧实验结果改称微调模型的端到端成绩。

报告同时说明 R4 是未微调混合链加重排，现有数据没有比较“微调 Dense + Cross-Encoder”。不运行重排是当前微调链的实现选择，不能据此证明再次重排在所有任务上都无收益。

为避免误解，检索实验的 A–D 在本报告中改为 R1–R4；全系统的 A/B/C 分别表示全局检索、分领域检索后共同生成、分领域独立生成后综合。检索的 400 条比较来自官方发布的 dummy test 文件，因此表述为开发比较。BGE 检索实验与 MiniLM 下游实验采用不同后端，报告没有将其当作同一条端到端增益。

## 阅读与编辑

按顺序修改 `sections/01_...tex` 至 `08_...tex`。`03_retrieval.tex` 由 `03_method.tex` 引入，检索队友可连同 `05_retrieval_existing.tex` 单独维护。作者目前采用团队署名 `HKUST NLP Project Team`；提交前可在 `main.tex` 中替换为真实成员姓名、学号或课程要求的署名。

本报告使用仓库已有实验结果，不新增付费 API 实验，不修改项目代码和历史结果。HotpotQA 尚无已完成实验，因此未给出其结果。未以自动评分声称幻觉率降低或任务完成率提高。独立人工审核和同样本完整消融仍属后续工作。

## 编译

可直接把本文件夹内的文件上传到 Overleaf，主文档选择 `main.tex`，编译器选择 XeLaTeX（也可使用 pdfLaTeX），执行完整的 BibTeX 编译流程。

报告已用 Tectonic 编译；本机临时编译器和缓存不入仓库。普通 LaTeX 环境可在本目录运行：

```text
xelatex main
bibtex main
xelatex main
xelatex main
```

`acl.sty` 与 `acl_natbib.bst` 为未修改的官方样式文件，来源提交见 `SOURCE.md`。未调整官方页面边距、正文大小或双栏规格。

## 图与结果来源

`figures/architecture.pdf` 为当前可选原生微调配置的流程图，检索框直接使用 fine-tuned BGE Dense；混合链与重排在正文中作为另一条配置及历史比较说明。`figures/coverage_cost.pdf` 为已有结果的覆盖与成本图。两图都是矢量图，并已包含在源码中，编译论文无需执行 Python。

`make_figures.py` 可用 Python + ReportLab 重建图；Windows 上会嵌入本机 Times 字体，其他系统会回退到 ReportLab 的 Times 字体。脚本仅绘图，不调用模型。

`evidence_manifest.json` 记录输入报告及结果文件的仓库相对路径与 SHA-256；提交文件哈希与原工作区哈希分开保留，避免Windows换行转换造成混淆。`EVIDENCE.md` 将表格和个案映射回项目结果文件。它们用于核对，不属于论文页数，也无需上传到课程系统。
