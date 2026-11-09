# 项目实验报告

本目录保存2026年11月9日整理的报告、可编辑源码和中文说明。

- [ACL 英文总报告](customer_service_rag_report.pdf)：8页，含参考文献和 Limitations，已完成编译及逐页排版检查。
- [LaTeX 源码](latex/)：主文件为 `main.tex`，可上传 Overleaf；编译方式和章节分工见其中的 [README](latex/README.md)。
- [中文说明与待完成工作](customer_service_rag_report_notes_zh.docx)：说明工程完成状态、实验数据来源、评价限制及补实验优先级。
- [队友的检索原稿](retrievalreport_en.docx)：原始文件未修改；方法、三张实验表和成本分析已整合进英文报告。

当前原生微调配置使用 struct + fine-tuned BGE Dense 直接排序，不执行 BM25、RRF 或 Cross-Encoder。报告保留的重排方法和 R4 属于另一条未微调混合对照；历史1000条与20条下游结果使用 MiniLM 混合链，不能视作当前微调配置的端到端成绩。

报告沿用仓库已有实验，没有新增模型运行。dummy400仅作历史开发比较；20条跨领域问题和自动评分仍待独立人工复核。HotpotQA适配、同样本完整消融和真实人工客服交接尚未完成。

数据与个案来源见 [EVIDENCE.md](latex/EVIDENCE.md)。`evidence_manifest.json` 中的路径相对仓库根目录，当前SHA-256以Git提交中的文件字节为准，同时保留原工作区哈希供溯源；原始Word按二进制原样复制。
